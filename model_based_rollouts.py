"""Fit regression to observed transition dynamics and rollout trajectories."""
import argparse
import logging
import os
import sys

from data_utils import create_factorized_dataset
from data_utils import make_env
from data_utils import StateActionStateDataset
from dynamics_models import LinearModelBasedSelectBounce
from dynamics_models import NeuralModelBasedSelectBounce
from dynamics_models import SeededSelectBounce
from plot_utils import anim


if __name__ == "__main__":
  parser = argparse.ArgumentParser("CoDA generate data.")
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
  parser.add_argument('--results_dir',
                      type=str,
                      default='/tmp/model_based_rollouts',
                      help='Output directory.')

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
  logging.basicConfig(filename=os.path.join(FLAGS.results_dir, 'main.log'),
                      level=logging.INFO)
  # log to std err
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  logging.getLogger('').addHandler(console)

  config, env = make_env(num_sprites=FLAGS.num_sprites, seed=FLAGS.seed, 
    max_episode_length=FLAGS.max_episode_length, imagedim=FLAGS.imagedim)
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

  # sample model-based rollouts
  if FLAGS.model_type == 'linear':
    model = LinearModelBasedSelectBounce(tr, seed=FLAGS.seed)
  elif FLAGS.model_type == 'neural':
    model = NeuralModelBasedSelectBounce(tr, seed=FLAGS.seed)
  else:
    raise ValueError("Bad model type.")
  config2, env2 = make_env(num_sprites=FLAGS.num_sprites, action_space=model, seed=FLAGS.seed, 
    max_episode_length=FLAGS.max_episode_length, imagedim=FLAGS.imagedim)

  # write movie of model-based rollouts
  res2 = anim(env2,
              FLAGS.num_frames,
              filename=os.path.join(FLAGS.results_dir, 'model_based.mp4'),
              **plot_kwargs)

  logging.info('done')
