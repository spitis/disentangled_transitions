"""Fit regression to observed transition dynamics and rollout trajectories."""
import argparse
import logging
import os
import pickle
import sys

import torch
from torch.utils.data import DataLoader

from data_utils import create_factorized_dataset
from data_utils import make_env
from data_utils import StateActionStateDataset
from dynamics_models import LinearModelBasedSelectBounce
from dynamics_models import LSTMModelBasedSelectBounce
from dynamics_models import NeuralModelBasedSelectBounce
from dynamics_models import compute_test_loss
from plot_utils import anim


if __name__ == "__main__":
  parser = argparse.ArgumentParser("Sample model-based rollouts.")
  parser.add_argument('--results_dir',
                      type=str,
                      default='/tmp/model_based_rollouts',
                      help='Output directory.')
  parser.add_argument('--num_sprites',
                      type=int,
                      default=4, help='Number of sprites.')
  parser.add_argument('--imagedim', 
                      type=int, 
                      default=16, 
                      help='Image dimension.')
  parser.add_argument('--num_examples',
                      type=int,
                      default=500,
                      help='Number of examples in dynamics model training set.')
  parser.add_argument('--num_frames',
                      type=int,
                      default=200,
                      help='Number of frames in sampled rollouts videos.')
  parser.add_argument('--seed',
                      type=int,
                      default=1,
                      help='Random seed.')
  parser.add_argument('--max_episode_length',
                      type=int,
                      default=5000,
                      help='Max length of an episode.')
  parser.add_argument('--model_type',
                      type=str,
                      default='linear',
                      help='Type of dynamics model.')
  parser.add_argument('--num_hidden_units',
                      type=int,
                      default=500,
                      help='Number of hidden units.')
  parser.add_argument('--lr',
                      type=float,
                      default=1e-3,
                      help='Learning rate.')
  parser.add_argument('--num_epochs',
                      type=int,
                      default=500,
                      help='Number of epochs.')
  parser.add_argument('--patience_epochs',
                      type=int,
                      default=20,
                      help='Stop early after this many epochs of '
                           'unimproved validation loss.')
  parser.add_argument('--batch_size',
                      type=int,
                      default=64,
                      help='Batch size.')
  parser.add_argument('--weight_decay',
                      type=float,
                      default=1e-5,
                      help='Weight decay.')

  FLAGS = parser.parse_args()
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
  log_filename = os.path.join(FLAGS.results_dir, 'main.log')
  if os.path.exists(log_filename):
    os.remove(log_filename)
  logging.basicConfig(filename=log_filename, level=logging.INFO)
  # log to std err
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  logging.getLogger('').addHandler(console)

  ground_truth_kwargs = dict(num_sprites=FLAGS.num_sprites, seed=FLAGS.seed,
    max_episode_length=FLAGS.max_episode_length, imagedim=FLAGS.imagedim)
  config, env = make_env(**ground_truth_kwargs)
  env.action_space.seed(FLAGS.seed)  # reproduce randomness in action space

  # write movie of actual environment rollouts
  plot_kwargs = dict(show_resets=True, show_clicks=True)
  res = anim(env,
             FLAGS.num_frames,
             filename=os.path.join(FLAGS.results_dir, 'ground_truth.mp4'),
             **plot_kwargs)

  # sample dataset from environment
  data, sprites = create_factorized_dataset(env, FLAGS.num_examples)
  tr = StateActionStateDataset(data, sprites)
  va = StateActionStateDataset(
    *create_factorized_dataset(env, FLAGS.num_examples)
  )  # validation data
  tr_loader = DataLoader(tr, FLAGS.batch_size, shuffle=True)
  va_loader = DataLoader(va, FLAGS.batch_size, shuffle=True)

  # sample model-based rollouts
  logging.info('Declaring a %s model.' % FLAGS.model_type)
  if FLAGS.model_type == 'linear':
    model = LinearModelBasedSelectBounce(tr, seed=FLAGS.seed)
  elif FLAGS.model_type == 'neural':
    activ_fn = torch.nn.ReLU()  # TODO(): replace with proper gin-configured fn
    model = NeuralModelBasedSelectBounce(tr_loader,
                                         va_loader,
                                         FLAGS.num_hidden_units,
                                         activ_fn,
                                         lr=FLAGS.lr,
                                         num_epochs=FLAGS.num_epochs,
                                         patience_epochs=FLAGS.patience_epochs,
                                         weight_decay=FLAGS.weight_decay,
                                         seed=FLAGS.seed)
  elif FLAGS.model_type == 'lstm':
    model = LSTMModelBasedSelectBounce(tr_loader,
                                       va_loader,
                                       lr=FLAGS.lr,
                                       num_epochs=FLAGS.num_epochs,
                                       patience_epochs=FLAGS.patience_epochs,
                                       weight_decay=FLAGS.weight_decay,
                                       seed=FLAGS.seed)
  else:
    raise ValueError("Bad model type.")

  # compute final test loss
  te = StateActionStateDataset(
    *create_factorized_dataset(env, FLAGS.num_examples)
  )  # test data
  te_loader = DataLoader(te)
  logging.info('final train loss: {:.7f}'.format(
    compute_test_loss(model, tr_loader)
  ))
  logging.info('final valid loss: {:.7f}'.format(
    compute_test_loss(model, va_loader)
  ))
  logging.info('final test loss: {:.7f}'.format(
    compute_test_loss(model, te_loader)
  ))

  model_based_kwargs = dict(num_sprites=FLAGS.num_sprites, action_space=model, seed=FLAGS.seed,
    max_episode_length=FLAGS.max_episode_length, imagedim=FLAGS.imagedim)
  config2, env2 = make_env(**model_based_kwargs)

  # write movie of model-based rollouts
  res2 = anim(env2,
              FLAGS.num_frames,
              filename=os.path.join(FLAGS.results_dir, 'model_based.mp4'),
              **plot_kwargs)

  # save environments kwargs to disk for reproducibility and use by RL agents
  with open(os.path.join(FLAGS.results_dir, 'ground_truth_kwargs.p'),
            'wb') as f:
    pickle.dump(ground_truth_kwargs, f)

  with open(os.path.join(FLAGS.results_dir, 'model_based_kwargs.p'), 'wb') as f:
    pickle.dump(model_based_kwargs, f)

  logging.info('done')
