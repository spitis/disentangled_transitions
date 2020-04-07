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

from absl_utils import log
from eval_utils import derive_acc, derive_f1
from structured_transitions import gen_samples_dynamic
from structured_transitions import MixtureOfMaskedNetworks, SimpleStackedAttn
from structured_transitions import TransitionsData

Array = np.ndarray
Tensor = torch.Tensor
DataLoader = torch.utils.data.DataLoader

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_attention_mechanism(
    train_loader: DataLoader,
    valid_loader: DataLoader,
    in_features: int,
    out_features: int,
    num_components: int,
    num_hidden_layers: int,
    num_hidden_units: int,
    lr: float,
    weight_decay: float,
    mask_reg: float,
    attn_reg: float,
    weight_reg: float,
    num_epochs: int,
    patient_epochs: int,
    tag: str = 'Training'
):
  del patient_epochs  # TODO(creager): use this for early stopping
  dev = 'cuda' if torch.cuda.is_available() else 'cpu'
  # build model and optimizer
  model_kwargs = dict(
    in_features=in_features,
    out_features=out_features,
    num_components=num_components,
    num_hidden_layers=num_hidden_layers,
    num_hidden_units=num_hidden_units
  )

  # A hack for now... 
  if in_features == 4:
    model = SimpleStackedAttn(**model_kwargs).to(dev)
  else:
    model = MixtureOfMaskedNetworks(**model_kwargs).to(dev)

  opt = torch.optim.Adam(model.parameters(), lr=lr,
                         weight_decay=weight_decay)
  pred_criterion = torch.nn.MSELoss()
  mask_criterion = torch.nn.L1Loss()
  auc_tr, auc_va = [], []
  losses_tr, losses_va = [], []

  for epoch in range(num_epochs):
    total_pred_loss, total_mask_loss, total_weight_loss, total_attn_loss = \
      (0., 0., 0., 0.)
    for i, (x, y, _) in enumerate(train_loader):
      pred_y, mask, attn = model.forward_with_mask(x.to(dev))
      pred_loss = pred_criterion(y.to(dev), pred_y)
      mask_loss = mask_reg * mask_criterion(
        torch.log(1. + mask), torch.zeros_like(mask))
      attn_loss = attn_reg * torch.sqrt(attn).mean()

      weight_loss = 0
      for param in model.parameters():
        weight_loss += mask_criterion(param, torch.zeros_like(param))
      weight_loss *= weight_reg

      total_pred_loss += pred_loss
      total_mask_loss += mask_loss
      total_attn_loss += attn_loss
      total_weight_loss += weight_loss

      loss = pred_loss + mask_loss + attn_loss + weight_loss
      model.zero_grad()
      loss.backward()
      opt.step()
    if epoch % 25 == 0:
      train_metrics = compute_metrics(model, train_loader)
      _, _, _, tr_auc, _, _ = train_metrics
      auc_tr.append(tr_auc)
      valid_metrics = compute_metrics(model, valid_loader)
      _, _, _, va_auc, _, _ = train_metrics
      auc_va.append(va_auc)
      losses_tr.append(loss.detach().item())
      # compute validation loss
      model.eval()
      valid_loss = 0.
      for i, (x, y, _) in enumerate(valid_loader):
        pred_y, mask, attn = model.forward_with_mask(x.to(dev))
        pred_loss = pred_criterion(y.to(dev), pred_y)
        mask_loss = mask_reg * mask_criterion(
          torch.log(1. + mask), torch.zeros_like(mask))
        attn_loss = attn_reg * torch.sqrt(attn).mean()

        weight_loss = 0
        for param in model.parameters():
          weight_loss += mask_criterion(param, torch.zeros_like(param))
        weight_loss *= weight_reg

        valid_loss += pred_loss + mask_loss + attn_loss + weight_loss
      valid_loss /= len(valid_loader)
      valid_loss = valid_loss.detach().item()
      losses_va.append(valid_loss)
      log(
        tag + ' ' +
        'Ep. {}. Pred: {:.5f}, Mask: {:.5f}, Attn: {:.5f}, Wt: {:.5f} '
        'Va loss {:.5f}. Tr AUC {:.5f}. Va AUC {:.5f}'
        .format(epoch, total_pred_loss / i, total_mask_loss / i,
               total_attn_loss / i, total_weight_loss / i,
                valid_loss, tr_auc, va_auc)
      )

  metrics = losses_tr, auc_tr, losses_va, auc_va
  model.eval()
  return model, model_kwargs, metrics


def local_model_sparsity(
      model, threshold: float, batch: Tuple[Tensor]
    ) -> Tuple[list, list]:
    x, _, ground_truth_sparsity = batch

    _, mask, _ = model.forward_with_mask(x.to(DEV))
    mask[mask < threshold] = 0
    mask = torch.where(mask > threshold,
                       torch.ones_like(mask),
                       torch.zeros_like(mask))
    predicted_sparsity = mask
    return predicted_sparsity, ground_truth_sparsity


def compute_metrics(
    model, loader: DataLoader
):
  model.eval()
  scores = []
  labels = []
  for x, _, m_tru in loader:
    _, m_hat, _ = model.forward_with_mask(x.to(DEV))
    scores.append(m_hat.detach().cpu().numpy().ravel())
    labels.append(m_tru.detach().cpu().numpy().ravel())
  scores = np.hstack(scores)
  labels = np.hstack(labels)
  fpr, tpr, thresh = metrics.roc_curve(labels, scores)
  auc = metrics.auc(fpr, tpr)
  acc = derive_acc(fpr, tpr, labels)
  f1 = derive_f1(fpr, tpr, labels)
  return fpr, tpr, thresh, auc, acc, f1


def plot_roc(
    results_dir: str,
    model,
    loader: DataLoader,
    tag_number: int = 0
):
  fpr, tpr, thresh, auc, acc, f1 = compute_metrics(model, loader)

  with open(os.path.join(results_dir,
                         'metrics_{}.p'.format(tag_number)), 'wb') as f:
    pickle.dump(dict(fpr=list(fpr),
                     tpr=tpr,
                     thresh=thresh,
                     auc=auc,
                     acc=acc,
                     f1=f1),
                f)

  import matplotlib
  matplotlib.use('Agg')
  from matplotlib import pyplot as plt

  plt.figure()
  lw = 2
  plt.plot(fpr, tpr, color='darkorange',
           lw=lw, label='ROC curve (area = %0.2f)' % auc)
  plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('ROC For Ground Truth Sparsity Recovery')
  plt.legend(loc="lower right")
  plt.savefig(os.path.join(results_dir, 'roc_{}.pdf'.format(tag_number)))

  plt.figure()
  lw = 2
  plt.plot(thresh, acc, color='darkorange', lw=lw, label='Acc(thresh)')
  # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
  # plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('Thresh')
  plt.ylabel('Accuracy')
  plt.title('Accuracy as fn of mask threshold.')
  plt.legend(loc="lower right")
  plt.savefig(os.path.join(results_dir, 'acc_{}.pdf'.format(tag_number)))

  plt.figure()
  lw = 2
  plt.plot(thresh, f1, color='darkorange', lw=lw, label='F1(thresh)')
  # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
  # plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('Thresh')
  plt.ylabel('F1')
  plt.title('F1 as fn of mask threshold.')
  plt.legend(loc="lower right")
  plt.savefig(os.path.join(results_dir, 'f1_{}.pdf'.format(tag_number)))

def plot_metrics(results_dir: str,
                 train_losses: list, train_aucs: list,
                 valid_losses: list, valid_aucs: list,
                 tag_number: int):
  import matplotlib
  matplotlib.use('Agg')
  from matplotlib import pyplot as plt

  # TODO(creager): proper x-axis: multiply ticks by 25

  fig, ax = plt.subplots(2, figsize=(8, 6))
  ax[0].plot(train_losses, linestyle='-', c='gray', label='Train loss')
  ax[1].plot(train_aucs, linestyle='-', c='blue', label='Train AUC')

  # plt.ylim(0.0003, 0.0015)
  ax[0].plot(valid_losses, linestyle='--', c='gray', label='Valid loss')
  ax[1].plot(valid_aucs, linestyle='--', c='blue', label='Valid AUC')

  ax[0].set_ylabel('Task loss', size=16)
  ax[1].set_ylabel('Attention AUC', size=16)
  for ax_ in ax:
    ax_.set_xlabel('25s of Epochs', size=16)
    ax_.legend()
  plt.suptitle('Learning metrics', size=16)
  plt.tight_layout()
  plt.savefig(os.path.join(results_dir, 'train_metrics_{}.pdf'.format(
    tag_number)))


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
  TAUS = np.linspace(0., .25, 11)  # tresholds to sweep when computing metrics

  results = dict(
    precision=defaultdict(list), recall=defaultdict(list)
  )

  FLAGS.splits = [int(split) for split in FLAGS.splits]

  for run in range(FLAGS.num_runs):
    seed = FLAGS.seed + run
    np.random.seed(seed)

    # create observational data
    global_interactions, fns, samples = gen_samples_dynamic(
      num_seqs=FLAGS.num_seqs, seq_len=FLAGS.seq_len, splits=FLAGS.splits,
      epsilon=FLAGS.epsilon)
    log('Total global interactions: {}/{}'
        .format(global_interactions, len(samples[0])))
    dataset = TransitionsData(samples)
    tr = TransitionsData(dataset[:int(len(dataset)*4/6)])
    va = TransitionsData(dataset[int(len(dataset)*4/6):int(len(dataset)*5/6)])
    te = TransitionsData(dataset[int(len(dataset)*5/6):])

    train_loader = torch.utils.data.DataLoader(tr, batch_size=FLAGS.batch_size,
                                               shuffle=True, num_workers=2,
                                               drop_last=True)
    valid_loader = torch.utils.data.DataLoader(va, batch_size=FLAGS.batch_size,
                                               shuffle=True, num_workers=2,
                                               drop_last=True)
    test_loader = torch.utils.data.DataLoader(te, batch_size=FLAGS.batch_size,
                                              shuffle=False, num_workers=2,
                                              drop_last=True)

    # train
    in_features = sum(FLAGS.splits)
    out_features = sum(FLAGS.splits)
    num_components = len(FLAGS.splits)  # TODO: make separate flag
    num_hidden_layers = 2  # TODO: make command line arg
    num_hidden_units = 256  # TODO: make command line arg
    patience_epochs = None  # TODO: make command line arg
    model, _, train_and_valid_metrics = train_attention_mechanism(
      train_loader,
      valid_loader,
      in_features,
      out_features,
      num_components,
      num_hidden_layers,
      num_hidden_units,
      FLAGS.lr,
      FLAGS.weight_decay,
      FLAGS.mask_reg,
      FLAGS.attn_reg,
      FLAGS.weight_reg,
      FLAGS.num_epochs,
      patience_epochs,
      tag='Run %d' % run
    )
    losses_tr, auc_tr, losses_va, auc_va = train_and_valid_metrics

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
      ground_truth_sparsity = torch.cat(ground_truth_sparsity).cpu().numpy()
      predicted_sparsity = torch.cat(predicted_sparsity).cpu().numpy()
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

    # plot train metrics
    plot_metrics(FLAGS.results_dir, losses_tr, auc_tr, losses_va, auc_va, run)

    # plot ROC
    plot_roc(FLAGS.results_dir, model, test_loader, run)

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

  log('results:\n' + pformat(results, indent=2))

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
  log('tex table:\n' + results_tex)

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

  log('done')


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
  flags.DEFINE_integer('num_runs', 10, 'Number of times to run the experiment.')
  flags.DEFINE_integer('seed', 1, 'Random seed.')
  flags.DEFINE_integer('num_epochs', 250, 'Number of epochs of training.')
  flags.DEFINE_list('splits', [3, 3, 3], 'Dimensions per state factor.')
  flags.DEFINE_boolean('verbose', False, 'If True, prints log info to std out.')
  flags.DEFINE_float('epsilon', 1.5, 'Sparse/dense threshold per factor in MP.')
  flags.DEFINE_string(
    'results_dir', '/tmp/dynamic_scm_discovery', 'Output directory.')

  app.run(main)
