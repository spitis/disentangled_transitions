"""Fit regression to observed transition dynamics and rollout trajectories."""
import argparse
import logging
import os
import sys

from colour import Color
import numpy as np
from spriteworld import environment, renderers, tasks
from spriteworld import factor_distributions as distribs
from spriteworld import sprite_generators
from spriteworld import gym_wrapper as gymw

from data_utils import create_factorized_dataset
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

  # build factors
  factors = distribs.Product([
    distribs.Continuous('x', 0.05, 0.95),
    distribs.Continuous('y', 0.05, 0.95),
    distribs.Continuous('c0', 25, 230),
    distribs.Continuous('c1', 25, 230),
    distribs.Continuous('c2', 25, 230),
    distribs.Continuous('x_vel', -0.08, 0.08),
    distribs.Continuous('y_vel', -0.08, 0.08),
    distribs.Discrete('shape', ['square']),
    distribs.Discrete('move_noise', [0.]),
    distribs.Discrete('scale', [0.15]),
  ])

  sprite_gen = sprite_generators.generate_nonintersecting_sprites(
    factors, num_sprites=FLAGS.num_sprites)
  sprite_gen = sprite_generators.sort_by_color(sprite_gen)

  # Above code produces random colors but has sensible ordering.
  # Below line forces fixed colors (bad for generalization, but presumably
  # easier to learn from images)

  # fix colors
  gradient_colors = list(Color("red").range_to(Color("blue"),
                                               FLAGS.num_sprites))
  gradient_colors = [
    tuple((np.array(gradient_color.get_rgb()) * 255).astype(np.int_))
    for gradient_color in gradient_colors
  ]
  sprite_gen = sprite_generators.fix_colors(sprite_gen,
                                            gradient_colors)

  random_mtx = (np.random.rand(100, 100) - 0.5) * 2.
  fn = lambda a: np.dot(random_mtx[:len(a), :len(a)], a)

  # WARNING: Because this uses velocity, using images makes it a POMDP!

  rndrs = {
    'image': renderers.PILRenderer(image_size=(FLAGS.imagedim, FLAGS.imagedim),
                                   anti_aliasing=16),
    'disentangled': renderers.VectorizedPositionsAndVelocities(),
    'entangled': renderers.FunctionOfVectorizedPositionsAndVelocities(fn=fn),
    'mask': renderers.TransitionEntanglementMask(state_size=4, action_size=2),
    'mask_abstract': renderers.TransitionEntanglementMask(state_size=1,
                                                          action_size=1)
  }

  # sample actual environment rollouts
  config = {
    'task': tasks.NoReward(),
    # 'action_space': action_spaces.SelectBounce(),
    'action_space': SeededSelectBounce(FLAGS.seed),
    'renderers': rndrs,
    'init_sprites': sprite_gen,
    'max_episode_length': FLAGS.max_episode_length,
    'metadata': {
      'name': 'test',  # os.path.basename(__file__),
    },
    'seed': FLAGS.seed
  }

  env = environment.Environment(**config)
  env = gymw.GymWrapper(env)
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
  config2 = {
    'task': tasks.NoReward(),
    'action_space': model,
    'renderers': rndrs,
    'init_sprites': sprite_gen,
    'max_episode_length': FLAGS.max_episode_length,
    'metadata': {
      'name': 'test',  # os.path.basename(__file__),
    },
    'seed': FLAGS.seed
  }

  env2 = environment.Environment(**config2)
  env2 = gymw.GymWrapper(env2)
  env2.action_space.seed(FLAGS.seed)  # reproduce randomness in action space

  # write movie of model-based rollouts
  res2 = anim(env2,
              FLAGS.num_frames,
              filename=os.path.join(FLAGS.results_dir, 'model_based.mp4'),
              **plot_kwargs)

  logging.info('done')
