from collections import defaultdict
from functools import partial
import json
import os
import pickle
from pprint import pformat
import sys

from absl import app
from absl import flags
from absl import logging
import numpy as np
import pandas as pd
from sklearn import metrics
import torch

from absl_utils import log
from coda import get_true_flat_mask
from data_utils import create_factorized_dataset
from data_utils import make_env
from data_utils import SpriteMaker
from data_utils import StateActionStateDataset
from dynamic_scm_discovery import local_model_sparsity
from dynamic_scm_discovery import plot_roc
from dynamic_scm_discovery import plot_metrics
from dynamic_scm_discovery import train_attention_mechanism
from structured_transitions import TransitionsData


Array = np.ndarray
Tensor = torch.Tensor
DataLoader = torch.utils.data.DataLoader

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def main(argv):
  """Train attention mechanism on state-action tuples from spriteworld env."""
  del argv  # unused

  log(FLAGS.results_dir)

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
  logging.get_absl_handler().use_absl_log_file('spriteworld_scm_discovery',
                                               FLAGS.results_dir)

  # TAUS = np.linspace(0., .5, 11)  # tresholds to sweep when computing metrics
  TAUS = np.linspace(0., .25, 11)  # tresholds to sweep when computing metrics

  results = dict(
    precision=defaultdict(list), recall=defaultdict(list)
  )
  dev = 'cuda' if torch.cuda.is_available() else 'cpu'

  for run in range(FLAGS.num_runs):
    auc_tr, auc_va = [], []
    losses_tr, losses_va = [], []
    seed = FLAGS.seed + run
    np.random.seed(seed)

    # create observational data
    ground_truth_kwargs = dict(num_sprites=FLAGS.num_sprites, seed=seed,
      max_episode_length=FLAGS.max_episode_length, imagedim=FLAGS.imagedim)
    config, env = make_env(**ground_truth_kwargs)
    env.action_space.seed(FLAGS.seed)  # reproduce randomness in action space

    sprite_maker = SpriteMaker(partial(make_env, **ground_truth_kwargs))

    # sample dataset from environment
    data, sprites = create_factorized_dataset(env, FLAGS.num_examples)
    tr = StateActionStateDataset(data, sprites)

    # build ground truth masks
    ground_truth_masks = []
    for s, a in zip(tr.s1, tr.a):
      # print(s.shape, a.shape)
      a = a.numpy()
      s = s.numpy()
      sprites_ = sprite_maker(s)
      mask = get_true_flat_mask(sprites_, config, a)
      # delete last to columns of the sparsity pattern, which represent the
      # mapping of current_state, current_action -> next_action
      mask = mask[:, :-2]
      ground_truth_masks.append(mask)
    ground_truth_masks = np.stack(ground_truth_masks)
    ground_truth_masks = torch.tensor(ground_truth_masks)

    # stack (s, a) tuples and build samples tuple consumable by TransitionsData
    states = tr.s1.reshape(FLAGS.num_examples, -1)
    actions = tr.a
    next_states = tr.s2.reshape(FLAGS.num_examples, -1)
    x = torch.cat((states, actions), -1)
    y = next_states
    m = ground_truth_masks
    samples = (x, y, m)

    dataset = TransitionsData(samples)
    tr = TransitionsData(dataset[:int(len(dataset)*4/6)])
    va = TransitionsData(dataset[int(len(dataset)*4/6):int(len(dataset)*5/6)])
    te = TransitionsData(dataset[int(len(dataset)*5/6):])

    train_loader = torch.utils.data.DataLoader(tr, batch_size=FLAGS.batch_size,
                                               shuffle=True, num_workers=2,
                                               drop_last=True)
    valid_loader = torch.utils.data.DataLoader(va, batch_size=FLAGS.batch_size,
                                               shuffle=False, num_workers=2,
                                               drop_last=True)
    test_loader = torch.utils.data.DataLoader(te, batch_size=FLAGS.batch_size,
                                              shuffle=False, num_workers=2,
                                              drop_last=True)

    # train
    num_action_features = 2
    num_state_features = FLAGS.num_sprites * 4
    if FLAGS.model_type == 'MMN':
      in_features = num_state_features + num_action_features
      out_features = num_state_features
      num_components = FLAGS.num_sprites  # TODO: make separate flag
      num_hidden_units = 256  # TODO: make command line arg
    elif FLAGS.model_type == 'SSA':
      in_features = 4
      out_features = 4
      num_components = 2  # TODO: make separate flag
      num_hidden_units = 512  # TODO: make command line arg
    else:
      msg = 'Unsupported model type. Got %s. Expected MMN or SSA.' % \
            FLAGS.model_type
      raise ValueError(msg)
    num_hidden_layers = 2  # TODO: make command line arg
    patience_epochs = None  # TODO: make command line arg
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
    plot_metrics(FLAGS.results_dir, losses_tr, auc_tr, losses_va, auc_va, seed)

    # plot ROC
    plot_roc(FLAGS.results_dir, model, test_loader, seed)

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

  # save trained model and its initializing kwargs for future use
  with open(os.path.join(FLAGS.results_dir, 'model_kwargs.json'), 'w') as f:
    f.write(json.dumps(model_kwargs, indent=2))

  model.to('cpu')
  torch.save(model.state_dict(), os.path.join(FLAGS.results_dir, 'model.p'))

  log('done')


if __name__ == "__main__":
  FLAGS = flags.FLAGS
  flags.DEFINE_integer('batch_size', 1000, 'Batch size.')
  flags.DEFINE_float('lr', 1e-3, 'Learining rate.')
  flags.DEFINE_float('mask_reg', 1e-3, 'Mask regularization coefficient.')
  flags.DEFINE_float('attn_reg', 1e-3, 'Attention regularization coefficient.')
  flags.DEFINE_float('weight_reg', 1e-3, 'Weight regularization coefficient.')
  flags.DEFINE_float('weight_decay', 1e-5, 'Weight decay.')
  # flags.DEFINE_integer('num_seqs', 1500, 'Number of sequences.')
  # flags.DEFINE_integer('seq_len', 10, 'Length of each sequence.')
  flags.DEFINE_integer('num_sprites', 4, 'Number of sprites.')
  flags.DEFINE_integer('imagedim', 16, 'Image dimension.')
  flags.DEFINE_integer('num_examples', 500, 'Number of examples in attention '
                                            'mechanism training set.')
  flags.DEFINE_integer('max_episode_length', 5000, 'Maximum length of an '
                                                   'episode.')
  flags.DEFINE_integer('num_runs', 10, 'Number of times to run the experiment.')
  flags.DEFINE_integer('seed', 1, 'Random seed.')
  flags.DEFINE_integer('num_epochs', 250, 'Number of epochs of training.')
  flags.DEFINE_integer('patience_epochs', 20, 'Stop early after this many '
                                              'epochs of unimproved '
                                              'validation loss.')
  flags.DEFINE_string('model_type', 'MMN', 'Type of attn mech.')
  # flags.DEFINE_list('splits', [3, 3, 3], 'Dimensions per state factor.')
  flags.DEFINE_boolean('verbose', False, 'If True, prints log info to std out.')
  flags.DEFINE_string(
    'results_dir', '/tmp/spriteworld_scm_discovery', 'Output directory.')

  app.run(main)
