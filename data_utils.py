import copy

import numpy as np
import torch
from colour import Color
from enum import Enum
import gym

import os, sys
sys.path.append(os.path.dirname('spritelu/'))
from spriteworld import environment, renderers, tasks
from spriteworld import factor_distributions as distribs
from spriteworld import sprite_generators
from spriteworld import gym_wrapper as gymw
from coda.dynamics_models import SeededSelectBounce, SeededSelectRedirect
from scipy.spatial.distance import pdist as pairwise_distance, squareform

class FlatEnvWrapper(gym.ObservationWrapper):
  """Flattens the environment observations so that only the disentangled observation is returned."""
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.observation_space = self.observation_space['disentangled']
    
  def observation(self, observation):
    return observation['disentangled'].flatten()

class SymmetricActionWrapper(gym.ActionWrapper):
  """Turns transforms action from (-1, 1) to (0, 1)."""
  def action(self, action):
    return (action + 1.) / 2.

class PairwiseDistanceSprites(tasks.AbstractTask):
  """Task is to min/max a function of pairwise distance between all the sprites"""
  
  def __init__(self, mode='max', fn=np.mean):
    if mode == 'max':
      self.coef = 1.
    elif mode == 'min':
      self.coef = -1.
    else:
      raise NotImplementedError
      
    self.fn = fn
    
  def reward(self, sprites):
    """Computes reward from list of sprites"""
    poses = [s.position for s in sprites]
    return self.coef * self.fn(pairwise_distance(poses))
    
    
  def reward_of_vector_repr(self, state_vector):
    """Computes reward on a 'VectorizedPositionsAndVelocities' format"""
    poses = state_vector.reshape(-1, 4)[:,:2]
    return self.coef * self.fn(pairwise_distance(poses))
    
  def success(self, sprites):
    return False # never terminates

class PlaceRewardType(Enum):
  SPARSE = 0   # 1 reward if placed all balls
  PARTIAL = 1  # 1/N reward for each ball placed on a target
  DENSE = 2    # Sum_N  1/N - 1/N*(max(0, Euclidean distance to target - Threshold))

MODES = dict(
  sparse=PlaceRewardType.SPARSE,
  partial=PlaceRewardType.PARTIAL,
  dense=PlaceRewardType.DENSE
)
class TargetGoalPos(tasks.AbstractTask):
  """Task is to move the first N sprites to N goal poses"""
  
  def __init__(self, N=4, mode=PlaceRewardType.SPARSE, eps=0.25):
    self.N = N
    self.targets = np.array([[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]])
    self.eps = eps
    self.mode = mode
    
  def reward(self, sprites):
    """Computes reward from list of sprites"""
    poses = np.array([s.position for s in sprites])
    return self.positions_to_reward(poses)
    
  def batch_reward_of_vector_repr(self, state_vector):
    """Computes reward on a 'VectorizedPositionsAndVelocities' format"""
    poses = state_vector.reshape(-1, 4, 4)[:, :,:2]
    return self.batch_positions_to_reward(poses)
  
  def batch_positions_to_reward(self, poses):
    dists = np.linalg.norm(poses - self.targets, axis=2)
    dists[:,:4-self.N] = 0
    if self.mode == PlaceRewardType.SPARSE:
      return float(np.all(dists < self.eps, axis=-1, keepdims=True))
    elif self.mode == PlaceRewardType.PARTIAL:
      return ((dists < self.eps).sum(-1, keepdims=True) - (4. - self.N)) / self.N
    elif self.mode == PlaceRewardType.DENSE:
      raise NotImplementedError
    
  def reward_of_vector_repr(self, state_vector):
    """Computes reward on a 'VectorizedPositionsAndVelocities' format"""
    poses = state_vector.reshape(-1, 4)[:,:2]
    return self.positions_to_reward(poses)
  
  def positions_to_reward(self, poses):
    dists = np.linalg.norm(poses - self.targets, axis=1)
    dists[:4-self.N] = 0
    if self.mode == PlaceRewardType.SPARSE:
      return float(np.all(dists < self.eps))
    elif self.mode == PlaceRewardType.PARTIAL:
      return ((dists < self.eps).sum() - (4. - self.N)) / self.N
    elif self.mode == PlaceRewardType.DENSE:
      rew = 0.
      for i in range(3, 3 - self.N, -1):
        rew += 1./self.N * (1. - max(0., dists[i] - self.eps))
      return rew

  def success(self, sprites):
    return False # never terminates


def make_env(num_sprites = 4, action_space = None, seed = 0,
  max_episode_length=50, imagedim=16, reward_type='min_pairwise'):

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
    factors, num_sprites=num_sprites)
  sprite_gen = sprite_generators.sort_by_color(sprite_gen)

  # Above code produces random colors but has sensible ordering.
  # Below line forces fixed colors (bad for generalization, but presumably
  # easier to learn from images)

  # fix colors
  gradient_colors = list(Color("red").range_to(Color("blue"),
                                               num_sprites))
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
    'image': renderers.PILRenderer(image_size=(imagedim, imagedim),
                                   anti_aliasing=16),
    'disentangled': renderers.VectorizedPositionsAndVelocities(),
    'entangled': renderers.FunctionOfVectorizedPositionsAndVelocities(fn=fn),
    'mask': renderers.TransitionEntanglementMask(state_size=4, action_size=2),
    'mask_abstract': renderers.TransitionEntanglementMask(state_size=1,
                                                          action_size=1)
  }
  
  if action_space is None:
    action_space = SeededSelectRedirect(seed)

  if reward_type == 'min_pairwise':
    task = PairwiseDistanceSprites('min')
  elif reward_type == 'max_pairwise':
    task = PairwiseDistanceSprites('max')
  elif 'place_' in reward_type:
    _, mode, N = reward_type.split('_')
    task = TargetGoalPos(mode=MODES[mode], N=int(N))
  else:
    raise NotImplementedError

  # sample actual environment rollouts
  config = {
    'task': task,
    # 'action_space': action_spaces.SelectBounce(),
    'action_space': action_space,
    'renderers': rndrs,
    'init_sprites': sprite_gen,
    'max_episode_length': max_episode_length,
    'metadata': {
      'name': 'test',  # os.path.basename(__file__),
    },
    'seed': seed
  }

  env = environment.Environment(**config)
  env = gymw.GymWrapper(env)
  assert(np.all(env.action_space.high == 1))
  assert(np.all(env.action_space.low == 0))
  env = SymmetricActionWrapper(env)
  env.action_space.low = -env.action_space.high
  env.action_space.seed(seed)  # reproduce randomness in action space
  return config, env
  

class SpriteMaker():
  def __init__(self, make_env=make_env):
    _, self.env = make_env()
    self.sprites = self.env.state()['sprites']
    
  def __call__(self, state):
    for sprite, s in zip(self.sprites, state.reshape(-1, 4)):
      sprite._position = s[:2]
      sprite._velocity = s[2:]
    
    return copy.deepcopy(self.sprites)    


def create_factorized_dataset(env, num_transitions=20000, reset_prob=0.05,
                              print_every=1000):
  data = []
  sprites = []
  s1 = env.reset()
  sprites1 = copy.deepcopy(env._env.state()['sprites'])
  i = 1
  while len(data) < num_transitions:
    i += 1
    if i % print_every == 0:
      print('.', end='', flush=True)
    a = env.action_space.sample()
    s2, r, _, _ = env.step(a)

    data.append((s1['disentangled'], a, r, s2['disentangled']))
    sprites.append(sprites1)

    s1 = s2
    sprites1 = copy.deepcopy(env._env.state()['sprites'])

    if np.random.random() < reset_prob:
      s1 = env.reset()
      sprites1 = copy.deepcopy(env._env.state()['sprites'])
  return data, sprites


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
    return s1, a, s2

