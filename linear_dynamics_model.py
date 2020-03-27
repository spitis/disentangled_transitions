"""Fit regression to observed transition dynamics and rollout trajectories."""
import argparse
import logging
import os
import sys

from colour import Color
import numpy as np
from sklearn.linear_model import LinearRegression
from spriteworld import environment, renderers, tasks, action_spaces
from spriteworld import factor_distributions as distribs
from spriteworld import sprite_generators
from spriteworld import gym_wrapper as gymw
import torch

from coda_forward_model import create_factorized_dataset
from utils import anim


class StateActionStateDataset(torch.utils.data.Dataset):
  """Relabels the data up front using relabel_strategy"""

  def __init__(self, data, sprites):
    self.data = data
    self.sprites = sprites

    self.s1, self.a, _, self.s2 = list(zip(*self.data))

    self.s1 = torch.tensor(self.s1).detach()
    self.a = torch.tensor(self.a).detach()
    self.s2 = torch.tensor(self.s2).detach()

  def __len__(self):
    return len(self.s1)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    s1 = self.s1[idx]
    a = self.a[idx]
    s2 = self.s2[idx]
    return torch.cat((s1, a), 0), s2


class StateActionTestDataset(torch.utils.data.Dataset):
  def __init__(self, data, sprites):
    self.s1, self.a, _, self.s2 = list(zip(*data))
    self.s1 = torch.tensor(self.s1).detach().flatten(start_dim=1)
    self.a = torch.tensor(self.a).detach()
    self.s2 = torch.tensor(self.s2).detach().flatten(start_dim=1)

  def __len__(self):
    return len(self.s1)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    s1 = self.s1[idx]
    a = self.a[idx]
    s2 = self.s2[idx]
    return torch.cat((s1, a), 0), s2


class ModelBasedSelectBounce(action_spaces.SelectBounce):
  """Swaps spriteworld environment dynamics with regressor learned from data."""
  def __init__(self, dataset, noise_scale=0.01, prevent_intersect=0.1):
    super(ModelBasedSelectBounce, self).__init__(
      noise_scale=noise_scale, prevent_intersect=prevent_intersect)
    # fit model
    data = dataset.data
    X = np.vstack(
      [np.hstack((state.ravel(), action.ravel()))
       for state, action, _, _ in data]
    )
    y = np.stack(
      [next_state.ravel() for _, _, _, next_state in data]
    )
    self.model = LinearRegression().fit(X, y).predict

  def step(self, action, sprites, *unused_args, **unused_kwargs):
    """Take an action and move the sprites.
    Args:
      action: Numpy array of shape (2,) in [0, 1].
      sprites: Iterable of sprite.Sprite() instances.
    Returns:
      Scalar cost of taking this action.
    """
    state = np.vstack(
      [np.hstack((sprite.x, sprite.y, sprite.velocity))
       for sprite in sprites]
    )  # N_sprites x 4 matrix
    model_input = np.hstack((state.ravel(), action.ravel()))  # flatten + concat
    model_input = model_input.reshape(1, -1)  # expand dimension for sklearn
    next_state = self.model(model_input)  # flattened predicted next state
    next_state = next_state.reshape(state.shape)  # unflatten
    for sprite, sprite_next_state in zip(sprites, next_state):
      position, velocity = sprite_next_state[:2], sprite_next_state[2:]
      sprite._position = position
      sprite._velocity = velocity

    return 0.


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
  parser.add_argument('--seed',
                      type=int,
                      default=1,
                      help='Random seed.')
  parser.add_argument('--max_episode_length',
                      type=int,
                      default=5000,
                      help='Max length of an episode.')
  parser.add_argument('--results_dir',
                      type=str,
                      default='/tmp/linear_dynamics_model',
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

  # set random seed
  np.random.seed(FLAGS.seed)

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
    'action_space': action_spaces.SelectBounce(),
    'renderers': rndrs,
    'init_sprites': sprite_gen,
    'max_episode_length': FLAGS.max_episode_length,
    'metadata': {
      'name': 'test',  # os.path.basename(__file__),
    }
  }

  env = environment.Environment(**config)
  env = gymw.GymWrapper(env)

  # write movie of actual environment rollouts
  res = anim(env, 200, filename=os.path.join(FLAGS.results_dir,
                                             'ground_truth.mp4'))

  # sample dataset from environment
  data, sprites = create_factorized_dataset(env, FLAGS.num_examples)
  tr = StateActionStateDataset(data, sprites)

  # sample model-based rollouts
  model =  ModelBasedSelectBounce(tr)
  config2 = {
    'task': tasks.NoReward(),
    'action_space': model,
    'renderers': rndrs,
    'init_sprites': sprite_gen,
    'max_episode_length': FLAGS.max_episode_length,
    'metadata': {
      'name': 'test',  # os.path.basename(__file__),
    }
  }

  env2 = environment.Environment(**config2)
  env2 = gymw.GymWrapper(env2)

  # write movie of model-based rollouts
  res2 = anim(env2, 200, filename=os.path.join(FLAGS.results_dir,
                                               'model_based.mp4'))

  logging.info('done')
