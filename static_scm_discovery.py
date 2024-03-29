from collections import defaultdict
import json
import os
import pickle
from pprint import pformat
import sys
from typing import Tuple
from typing import Iterable

from absl import app
from absl import flags
from absl import logging
import numpy as np
import pandas as pd
import scipy
from sklearn import metrics
import torch

from dynamic_scm_discovery import train_attention_mechanism
from dynamic_scm_discovery import plot_metrics
from dynamic_scm_discovery import plot_roc
from structured_transitions import gen_samples_static
from structured_transitions import MaskedNetwork
from structured_transitions import TransitionsData

Array = np.ndarray


def model_sparsity(model: MaskedNetwork, threshold: float) -> Array:
    assert isinstance(model, MaskedNetwork), 'bad model'
    mask = model.mask.detach().cpu().numpy()
    mask[mask < threshold] = 0
    mask = (mask > 0).astype(np.int_)
    return mask


def precision_recall(
    model: MaskedNetwork, threshold: float, splits: Iterable
) -> Tuple[float, float]:
  predicted_sparsity = model_sparsity(model, threshold)
  ground_truth_sparsity = scipy.linalg.block_diag(*(
        np.ones((split, split), dtype=np.int_) for split in splits
  ))
  precision = metrics.precision_score(
      ground_truth_sparsity.ravel(),
      predicted_sparsity.ravel()
  )
  recall = metrics.recall_score(
      ground_truth_sparsity.ravel(),
      predicted_sparsity.ravel()
  )
  return precision, recall


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
  logging.get_absl_handler().use_absl_log_file('static_scm_discovery',
                                               FLAGS.results_dir)

  TAUS = np.linspace(0., .15, 4)  # tresholds to sweep when computing metrics

  results = dict(
    precision=defaultdict(list), recall=defaultdict(list)
  )
  dev = 'cuda' if torch.cuda.is_available() else 'cpu'

  FLAGS.splits = [int(split) for split in FLAGS.splits]



  for run in range(FLAGS.num_runs):
    logging.info('Run %d' % run)
    np.random.seed(FLAGS.seed + run)

    # create observational data
    fns, samples = gen_samples_static(
      num_seqs=FLAGS.num_seqs, seq_len=FLAGS.seq_len, splits=FLAGS.splits)
    dataset = TransitionsData(samples)
    tr = TransitionsData(dataset[:int(len(dataset)*4/6)])
    va = TransitionsData(dataset[int(len(dataset)*4/6):int(len(dataset)*5/6)])
    te = TransitionsData(dataset[int(len(dataset)*5/6):])

    train_loader = torch.utils.data.DataLoader(
      tr, batch_size=FLAGS.batch_size, shuffle=True, num_workers=2,
      drop_last=True)
    valid_loader = torch.utils.data.DataLoader(
      va, batch_size=FLAGS.batch_size, shuffle=True, num_workers=2,
      drop_last=True)
    test_loader = torch.utils.data.DataLoader(
      te, batch_size=FLAGS.batch_size, shuffle=False, num_workers=2,
      drop_last=True)
    logging.info('Data created.')



    num_action_features = 0
    num_state_features = sum(FLAGS.splits)
    num_sprites = len(FLAGS.splits)  # really num_subspaces
    if FLAGS.model_type == 'MMN':
      in_features = num_state_features + num_action_features
      out_features = num_state_features
      num_components = num_sprites  # TODO: make separate flag
      num_hidden_units = 256  # TODO: make command line arg
      attn_reg = 1e-3  # TODO: make proper command line arg
      weight_reg = 1e-3  # TODO: make proper command line arg
      mask_reg = 1e-3  # TODO: make proper command line arg
    elif FLAGS.model_type == 'SSA':
      in_features = 4
      out_features = 4
      num_components = 2  # TODO: make separate flag
      num_hidden_units = 512  # TODO: make command line arg
      attn_reg = 0.  # TODO: make proper command line arg
      weight_reg = 0.  # TODO: make proper command line arg
      mask_reg = 0.  # TODO: make proper command line arg
    else:
      msg = 'Unsupported model type. Got %s. Expected MMN or SSA.' % \
            FLAGS.model_type
      raise ValueError(msg)

    # in_features = sum(FLAGS.splits)
    # out_features = sum(FLAGS.splits)
    # num_components = len(FLAGS.splits)  # TODO: make separate flag
    num_hidden_layers = 2  # TODO: make command line arg
    # num_hidden_units = 256  # TODO: make command line arg
    patience_epochs = None
    # attn_reg = 1e-3  # TODO: make proper command line arg
    # weight_reg = 1e-3  # TODO: make proper command line arg
    # mask_reg = 1e-3  # TODO: make proper command line arg
    model, model_kwargs, train_and_valid_metrics = train_attention_mechanism(
      train_loader,
      valid_loader,
      in_features,
      out_features,
      num_components,
      num_hidden_layers,
      num_hidden_units,
      FLAGS.lr,
      FLAGS.weight_decay,
      mask_reg,
      attn_reg,
      weight_reg,
      FLAGS.num_epochs,
      patience_epochs,
      tag='Run %d' % run
    )


    # # build model and optimizer
    # model = MaskedNetwork(in_features=sum(FLAGS.splits),
    #                       out_features=sum(FLAGS.splits),
    #                       num_hidden_layers=2, num_hidden_units=256).to(dev)
    # opt = torch.optim.Adam(model.parameters(), lr=FLAGS.lr,
    #                        weight_decay=FLAGS.weight_decay)
    # pred_criterion = torch.nn.MSELoss()
    # mask_criterion = torch.nn.L1Loss()
    #
    # # train
    # for epoch in range(FLAGS.num_epochs):
    #   total_pred_loss, total_mask_loss = 0., 0.
    #   for i, (x, y, _) in enumerate(train_loader):
    #     pred_y = model(x.to(dev))
    #     pred_loss = pred_criterion(y.to(dev), pred_y)
    #     mask = model.mask
    #     mask_loss = FLAGS.mask_reg * mask_criterion(
    #       torch.log(1. + mask), torch.zeros_like(mask))
    #
    #     total_pred_loss += pred_loss
    #     total_mask_loss += mask_loss
    #
    #     loss = pred_loss + mask_loss
    #     model.zero_grad()
    #     loss.backward()
    #     opt.step()
    #
    #   if epoch % 10 == 0:
    #     logging.info(
    #       'Run {} Epoch {} done! Pred loss: {:.5f}, Mask loss: {:.5f}'
    #       .format(run, epoch, total_pred_loss / i, total_mask_loss / i)
    #     )

    # plot train metrics
    losses_tr, auc_tr, losses_va, auc_va = train_and_valid_metrics
    plot_metrics(FLAGS.results_dir, losses_tr, auc_tr, losses_va, auc_va, run)

    # plot ROC
    plot_roc(FLAGS.results_dir, model, test_loader, run)

    # # eval
    # for tau in TAUS:
    #   precision, recall = precision_recall(model, tau, FLAGS.splits)
    #   results['precision'][tau].append(precision)
    #   results['recall'][tau].append(recall)

  # logging.info('results:\n' + pformat(results, indent=2))

  # format results as tex via pandas
  results_df = pd.DataFrame.from_dict(results)

  def format_mean_and_std(result):
    return '$%.2f \pm %.2f$' % (np.mean(result), np.std(result))

  # tau is inserted as column for improved formatting
  results_df.insert(0, r'$\tau$', TAUS.tolist())
  formatters = [lambda x: '%.2f' % x, format_mean_and_std, format_mean_and_std]
  results_tex = results_df.to_latex(
    formatters=formatters, escape=False, label='tab:static', index=False,
    column_format='|r|l|l|'
  )
  logging.info('tex table:\n' + results_tex)

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
  flags.DEFINE_integer('batch_size', 250, 'Batch size.')
  flags.DEFINE_float('lr', 1e-3, 'Learining rate.')
  flags.DEFINE_float('mask_reg', 2e-3, 'Mask regularization coefficient.')
  flags.DEFINE_float('weight_decay', 1e-4, 'Weight decay.')
  flags.DEFINE_integer('num_seqs', 600, 'Number of sequences.')
  flags.DEFINE_integer('seq_len', 10, 'Length of each sequence.')
  flags.DEFINE_integer('num_runs', 5, 'Number of times to run the experiment.')
  flags.DEFINE_integer('seed', 123, 'Random seed.')
  flags.DEFINE_integer('num_epochs', 100, 'Number of epochs of training.')
  flags.DEFINE_list('splits', [4, 3, 2], 'Dimensions per state factor.')
  flags.DEFINE_boolean('verbose', False, 'If True, prints log info to std out.')
  flags.DEFINE_string('model_type', 'MMN', 'Type of attn mech.')
  flags.DEFINE_string(
    'results_dir', '/tmp/static_scm_discovery', 'Output directory.')

  app.run(main)
