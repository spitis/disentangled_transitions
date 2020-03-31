import copy

import numpy as np
import torch


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

