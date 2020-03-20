from collections import defaultdict
import json
import os
import pickle
from pprint import pformat
import sys
from typing import Tuple

from absl import app
from absl import flags
from absl import logging
import numpy as np
import pandas as pd
from sklearn import metrics
import torch

from static_scm_discovery import precision_recall
from structured_transitions import gen_samples_dynamic
from structured_transitions import MixtureOfMaskedNetworks
from structured_transitions import TransitionsData

Array = np.ndarray
Tensor = torch.Tensor
DataLoader = torch.utils.data.DataLoader


def local_model_sparsity(
      model: MixtureOfMaskedNetworks, threshold: float, batch: Tuple[Tensor]
    ) -> Tuple[list, list]:
    x, _, ground_truth_sparsity = batch
    assert isinstance(model, MixtureOfMaskedNetworks), 'bad model'
    _, mask, _ = model.forward_with_mask(x)
    mask[mask < threshold] = 0
    mask = torch.where(mask > threshold,
                       torch.ones_like(mask),
                       torch.zeros_like(mask))
    predicted_sparsity = mask
    return predicted_sparsity, ground_truth_sparsity


def plot_roc(
    model: MixtureOfMaskedNetworks, loader: DataLoader
):
  import matplotlib
  matplotlib.use('Agg')
  from matplotlib import pyplot as plt
  scores = []
  labels = []
  for x, _, m_tru in loader:
    _, m_hat, _ = model.forward_with_mask(x)
    scores.append(m_hat.detach().numpy().ravel())
    labels.append(m_tru.detach().numpy().ravel())
  scores = np.hstack(scores)
  labels = np.hstack(labels)
  fpr, tpr, thresh = metrics.roc_curve(labels, scores)
  auc = metrics.auc(fpr, tpr)
  plt.figure()
  lw = 2
  plt.plot(fpr, tpr, color='darkorange',
           lw=lw, label='ROC curve (area = %0.2f)' % auc)
  plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver operating characteristic example')
  plt.legend(loc="lower right")
  plt.savefig(os.path.join(FLAGS.results_dir, 'roc.pdf'))


def main(argv):
  """Run the experiment."""
  del argv  # unused

  if not os.path.exists(FLAGS.results_dir):
    os.makedirs(FLAGS.results_dir)

  # for reproducibility, save command and script
  if FLAGS.results_dir is not '.':
    cmd = 'python ' + ' '.join(sys.argv)
    with open(os.path.join(FLAGS.results_dir, 'command.sh'), 'w') as f:
      f.write(cmd)
    this_script = open(__file__, 'r').readlines()
    with open(os.path.join(FLAGS.results_dir, __file__), 'w') as f:
      f.write(''.join(this_script))

  if not os.path.exists(FLAGS.results_dir):
    os.makedirs(FLAGS.results_dir)

  # set log file
  logging.get_absl_handler().use_absl_log_file('dynamic_scm_discovery',
                                               FLAGS.results_dir)

  # TAUS = np.linspace(0., .5, 11)  # tresholds to sweep when computing metrics
  TAUS = np.linspace(0., .25, 6)  # tresholds to sweep when computing metrics

  results = dict(
    precision=defaultdict(list), recall=defaultdict(list)
  )
  dev = 'cuda' if torch.cuda.is_available() else 'cpu'

  FLAGS.splits = [int(split) for split in FLAGS.splits]

  for run in range(FLAGS.num_runs):
    np.random.seed(FLAGS.seed + run)

    # create observational data
    global_interactions, fns, samples = gen_samples_dynamic(
      num_seqs=FLAGS.num_seqs, seq_len=FLAGS.seq_len, splits=FLAGS.splits,
      epsilon=FLAGS.epsilon)
    logging.info('Total global interactions: {}/{}'
                 .format(global_interactions, len(samples[0])))
    dataset = TransitionsData(samples)
    tr = TransitionsData(dataset[:int(len(dataset) * 5 / 6)])
    te = TransitionsData(dataset[int(len(dataset) * 5 / 6):])

    train_loader = torch.utils.data.DataLoader(tr, batch_size=FLAGS.batch_size,
                                               shuffle=True, num_workers=2,
                                               drop_last=True)
    test_loader = torch.utils.data.DataLoader(te, batch_size=FLAGS.batch_size,
                                              shuffle=False, num_workers=2,
                                              drop_last=True)

    # build model and optimizer
    model = MixtureOfMaskedNetworks(in_features=sum(FLAGS.splits),
                                    out_features=sum(FLAGS.splits),
                                    num_components=3,
                                    num_hidden_layers=2,
                                    num_hidden_units=256).to(dev)

    opt = torch.optim.Adam(model.parameters(), lr=FLAGS.lr,
                           weight_decay=FLAGS.weight_decay)
    pred_criterion = torch.nn.MSELoss()
    mask_criterion = torch.nn.L1Loss()

    # train

    for epoch in range(FLAGS.num_epochs):
      total_pred_loss, total_mask_loss, total_weight_loss, total_attn_loss = \
        (0., 0., 0., 0.)
      for i, (x, y, _) in enumerate(train_loader):
        pred_y, mask, attn = model.forward_with_mask(x.to(dev))
        pred_loss = pred_criterion(y.to(dev), pred_y)
        mask_loss = FLAGS.mask_reg * mask_criterion(
          torch.log(1. + mask), torch.zeros_like(mask))
        attn_loss = FLAGS.attn_reg * torch.sqrt(attn).mean()

        weight_loss = 0
        for param in model.parameters():
          weight_loss += mask_criterion(param, torch.zeros_like(param))
        weight_loss *= FLAGS.weight_reg

        total_pred_loss += pred_loss
        total_mask_loss += mask_loss
        total_attn_loss += attn_loss
        total_weight_loss += weight_loss

        loss = pred_loss + mask_loss + attn_loss + weight_loss
        model.zero_grad()
        loss.backward()
        opt.step()
      if epoch % 25 == 0:
        logging.info(
          'Run {}, Ep. {}. Pred: {:.5f}, Mask: {:.5f}, Attn: {:.5f}, Wt: {:.5f}'
          .format(run, epoch, total_pred_loss / i, total_mask_loss / i,
                 total_attn_loss / i, total_weight_loss / i))

    # worst-case eval w.r.t. actual dynamic masks
    for tau in TAUS:
      ground_truth_sparsity = []  # per step
      predicted_sparsity = []  # per step
      for batch in test_loader:
        predicted_sparsity_, ground_truth_sparsity_ = local_model_sparsity(
          model, tau, batch
        )
        ground_truth_sparsity.append(ground_truth_sparsity_)
        predicted_sparsity.append(predicted_sparsity_)
      ground_truth_sparsity = torch.cat(ground_truth_sparsity).numpy()
      predicted_sparsity = torch.cat(predicted_sparsity).numpy()
      results['precision'][tau].append(
        metrics.precision_score(
            ground_truth_sparsity.ravel(),
            predicted_sparsity.ravel()
        )
      )
      results['recall'][tau].append(
        metrics.recall_score(
            ground_truth_sparsity.ravel(),
            predicted_sparsity.ravel()
        )
      )


    # # best-case eval w.r.t. the splits hyperparams (not dynamic)
    # for tau in TAUS:
    #   # NOTE: we measure whether _any_ component uncovers the local transitions
    #   best_precision, best_recall = 0., 0.
    #   for component in model.components:  # best-case result across components
    #     precision, recall = precision_recall(component, tau, FLAGS.splits)
    #     if np.mean((precision, recall)) > np.mean((best_precision,
    #                                                best_recall)):
    #       best_precision, best_recall = precision, recall
    #   results['precision'][tau].append(best_precision)
    #   results['recall'][tau].append(best_recall)

  logging.info('results:\n' + pformat(results, indent=2))

  # format results as tex via pandas
  results_df = pd.DataFrame.from_dict(results)

  def format_mean_and_std(result):
    return '$%.2f \pm %.2f$' % (np.mean(result), np.std(result))

  # tau is inserted as column for improved formatting
  results_df.insert(0, r'$\tau$', TAUS.tolist())
  formatters = [lambda x: '%.2f' % x, format_mean_and_std, format_mean_and_std]
  results_tex = results_df.to_latex(
    formatters=formatters, escape=False, label='tab:dynamic', index=False,
    column_format='|r|l|l|'
  )
  logging.info('tex table:\n' + results_tex)

  # plot ROC
  plot_roc(model, test_loader)

  # save results (in various formats) to disk
  results.update(taus=TAUS.tolist())  # don't do this before pd.DataFrame init
  with open(os.path.join(FLAGS.results_dir, 'results.txt'), 'w') as f:
    f.write(pformat(results, indent=2))

  with open(os.path.join(FLAGS.results_dir, 'results.json'), 'w') as f:
    f.write(json.dumps(results, indent=2))

  with open(os.path.join(FLAGS.results_dir, 'results.tex'), 'w') as f:
    f.write(results_tex)

  with open(os.path.join(FLAGS.results_dir, 'results.p'), 'wb') as f:
    pickle.dump(results, f)

  logging.info('done')


if __name__ == "__main__":
  FLAGS = flags.FLAGS
  flags.DEFINE_integer('batch_size', 1000, 'Batch size.')
  flags.DEFINE_float('lr', 1e-3, 'Learining rate.')
  flags.DEFINE_float('mask_reg', 1e-3, 'Mask regularization coefficient.')
  flags.DEFINE_float('attn_reg', 1e-3, 'Attention regularization coefficient.')
  flags.DEFINE_float('weight_reg', 1e-3, 'Weight regularization coefficient.')
  flags.DEFINE_float('weight_decay', 1e-5, 'Weight decay.')
  flags.DEFINE_integer('num_seqs', 1500, 'Number of sequences.')
  flags.DEFINE_integer('seq_len', 10, 'Length of each sequence.')
  flags.DEFINE_integer('num_runs', 5, 'Number of times to run the experiment.')
  flags.DEFINE_integer('seed', 1, 'Random seed.')
  flags.DEFINE_integer('num_epochs', 250, 'Number of epochs of training.')
  flags.DEFINE_list('splits', [4, 3, 2], 'Dimensions per state factor.')
  flags.DEFINE_boolean('verbose', False, 'If True, prints log info to std out.')
  flags.DEFINE_float('epsilon', 1.5, 'Sparse/dense threshold per factor in MP.')
  flags.DEFINE_string(
    'results_dir', '/tmp/dynamic_scm_discovery', 'Output directory.')

  app.run(main)
