import argparse
import copy
from enum import Enum
from functools import reduce
from itertools import combinations, chain
import logging
import os
import pickle
import sys

import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
from scipy.sparse.csgraph import connected_components
from spriteworld import environment, renderers, tasks, action_spaces
from spriteworld import factor_distributions as distribs
from spriteworld import sprite_generators
from spriteworld import gym_wrapper as gymw
from matplotlib import animation
import torch

from structured_transitions import MaskedNetwork


def plot_losses(tr_loss_none, te_loss_none,
                tr_loss_rand, te_loss_rand,
                tr_loss_true, te_loss_true):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  import seaborn as sns
  sns.set(style='white')
  plt.figure(figsize=(8, 6))
  plt.plot(te_loss_none, linestyle='-', c='gray', label='No relabeling')
  plt.plot(te_loss_rand, linestyle='-', c='blue', label='Random relabeling')
  plt.plot(te_loss_true, linestyle='-', c='orange',
           label='Ground truth relabeling')

  plt.ylim(0.0003, 0.0015)
  plt.plot(tr_loss_none, linestyle='--', c='gray')
  plt.plot(tr_loss_rand, linestyle='--', c='blue')
  plt.plot(tr_loss_true, linestyle='--', c='orange')

  plt.legend()
  plt.ylabel('Mean squared error', size=16)
  plt.xlabel('Epochs', size=16)
  plt.title('Learning curves for different relabeling strategies', size=16)
  plt.show
  plt.savefig(os.path.join(FLAGS.results_dir, 'losses.pdf'))


class RelabelStrategy(Enum):
  NONE = 0
  GROUND_TRUTH = 1
  RANDOM = 2


def train_fwd_model(tr, te):
  dev = 'cuda' if torch.cuda.is_available() else 'cpu'
  train_loader = torch.utils.data.DataLoader(tr, batch_size=FLAGS.batch_size,
                                             shuffle=True,
                                             num_workers=2, drop_last=True)
  test_loader = torch.utils.data.DataLoader(te, batch_size=FLAGS.batch_size,
                                            shuffle=False, num_workers=2,
                                            drop_last=True)

  model = MaskedNetwork(in_features=FLAGS.num_sprites * 4 + 2,
                        out_features=FLAGS.num_sprites * 4,
                        num_hidden_layers=2, num_hidden_units=256).to(dev)
  opt = torch.optim.Adam(model.parameters(), lr=FLAGS.lr,
                         weight_decay=FLAGS.weight_decay)
  criterion = torch.nn.MSELoss()

  tr_losses = []
  te_losses = []
  for epoch in range(FLAGS.num_epochs):
    total_loss = 0.
    for i, (x, y) in enumerate(train_loader):
      pred_y = model(x.to(dev))
      loss = criterion(y.to(dev), pred_y)
      total_loss += loss
      model.zero_grad()
      loss.backward()
      opt.step()
    test_loss = 0.
    for j, (x, y) in enumerate(test_loader):
      pred_y = model(x.to(dev))
      loss = criterion(y.to(dev), pred_y)
      test_loss += loss
    tr_losses.append(float(total_loss / i))
    te_losses.append(float(test_loss / j))
    if epoch % 10 == 0:
      logging.info(
        'Epoch {} done! Train loss: {:.5f}.  Test loss: {:.5f}'
        .format(epoch, total_loss / i, test_loss / j)
      )

  return tr_losses, te_losses


def viz(obs, filename='./viz.pdf'):
  plt.figure(figsize=(2, 2))
  plt.imshow(255 - obs)
  plt.savefig(filename)


def anim(env, T=100):
  fig = plt.figure(figsize=(2, 2))

  states = [255 - env.reset()['image']]

  for i in range(T):
    a = env.action_space.sample()
    state, _, _, _ = env.step(a)
    states.append(255 - state['image'])

  im = plt.imshow(states[0], cmap=plt.get_cmap('jet'), vmin=0, vmax=255)

  def updatefig(j):
    im.set_array(states[j])
    return [im]

  ani = animation.FuncAnimation(fig, updatefig, frames=T, interval=75,
                                repeat_delay=1000)
  # Set up formatting for the movie files
  Writer = animation.writers['ffmpeg']
  writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
  ani.save(os.path.join(FLAGS.results_dir, 'env.mp4'), writer=writer)
  return ani.to_html5_video()


if __name__ == "__main__":
  parser = argparse.ArgumentParser("CoDA generate data.")
  parser.add_argument('--num_sprites',
                      type=int,
                      default=4, help='Number of sprites.')
  parser.add_argument('--imagedim', 
                      type=int, 
                      default=16, 
                      help='Image dimension.')
  parser.add_argument('--batch_size',
                      type=int,
                      default=1000, 
                      help='Batch size.')
  parser.add_argument('--lr',
                      type=float,
                      default=1e-3, 
                      help='Learining rate.')
  parser.add_argument('--weight_decay',
                      type=float, 
                      default=1e-5, 
                      help='Weight decay.')
  parser.add_argument('--num_pairs',
                      type=int,
                      default=500,
                      help='Number of transition pairs.')
  parser.add_argument('--relabel_samples_per_pair',
                      type=int,
                      default=10,
                      help='Number of relabels per transition pairs.')
  parser.add_argument('--seed',
                      type=int,
                      default=1,
                      help='Random seed.')
  parser.add_argument('--num_epochs',
                      type=int,
                      default=5,
                      help='Number of epochs of training.')
  parser.add_argument('--max_episode_length',
                      type=int,
                      default=5000,
                      help='Max length of an episode.')
  parser.add_argument('--results_dir',
                      type=str,
                      default='/tmp/coda_forward_model',
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
                      level=logging.DEBUG)

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

  sprite_gen = sprite_generators.fix_colors(sprite_gen,
                                            [(250, 125, 0), (0, 255, 125),
                                             (125, 0, 255), (255, 255, 255)])

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

  res = anim(env, 200)

  def create_factorized_dataset(num_transitions=20000, reset_prob=0.05,
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

  data, sprites = create_factorized_dataset(2000)
  test_data, test_sprites = create_factorized_dataset(5000)

  def powerset(n):
    xs = list(range(n))
    return list(chain.from_iterable(combinations(xs, n) for n in range(n + 1)))

  def get_true_abstract_mask(sprites, action=(0.5, 0.5)):
    """Returns a mask with iteractions for next transition given true sprites.
    E.g., returns [[1,0,0],[0,1,0],[0,0,1]] for 3 sprites"""
    sprites1 = copy.deepcopy(sprites)
    config['action_space'].step(action, sprites1)
    return config['renderers']['mask_abstract'].render(sprites1)

  def get_true_flat_mask(sprites, action=(0.5, 0.5)):
    """Returns a mask with iteractions for next transition given true sprites.
    E.g., returns [[1,0,0],[0,1,0],[0,0,1]] for 3 sprites"""
    sprites1 = copy.deepcopy(sprites)
    config['action_space'].step(action, sprites1)
    return config['renderers']['mask'].render(sprites1)

  def get_random_flat_mask(sprites, action=(0.5, 0.5)):
    sprites1 = copy.deepcopy(sprites)
    config['action_space'].step(action, sprites1)
    mask = config['renderers']['mask'].render(sprites1)
    return np.eye(len(mask))

  def get_fully_connected_mask(sprites, action=(0.5, 0.5)):
    sprites1 = copy.deepcopy(sprites)
    config['action_space'].step(action, sprites1)
    mask = config['renderers']['mask'].render(sprites1)
    return np.ones(mask.shape)

  def cc_list(cc):
    """Converts return of scipy's connected_components into a list of
    connected component indices tuples.
    E.g., if there are 4 nodes in the graph,
    this might return [array([0]), array([1]), array([2, 3])]
    """
    res = []
    num_ccs, cc_idxs = cc
    for i in range(num_ccs):
      res.append(np.where(cc_idxs == i)[0])
    return res

  def get_cc_from_sprites_and_action(sprites, action=(0.5, 0.5),
                                     get_mask=get_true_abstract_mask):
    """Returns list of connected component indices for next transition
    interactions given true sprites.

    E.g., if mask is [[1,0,0,0],[0,1,0,0],[0,0,1,1],[0,0,1,1]],
    this will return [array([0]), array([1]), array([2, 3])]
    """
    return cc_list(connected_components(get_mask(sprites, action)))

  def reduce_cc_list_by_union(cc_list, max_ccs):
    """Takes a cc list that is too long and merges some components to bring it
    to max_ccs"""
    while len(cc_list) > max_ccs:
      i, j = np.random.choice(range(1, len(cc_list) - 1), 2, replace=False)
      if (j == 0) or (j == len(cc_list) - 1):
        continue  # don't want to delete the base
      cc_list[i] = np.union1d(cc_list[i], cc_list[j])
      del cc_list[j]
    return cc_list

  def disentangled_components(cc_lst, max_components=5):
    """Converts connected component list into a list of disentangled subsets
    of the indices.
    """
    subsets = powerset(len(cc_lst))
    res = []
    for subset in subsets:
      res.append(
        reduce(np.union1d, [cc_lst[i] for i in subset], np.array([])).astype(
          np.int64))
    return list(map(tuple, res))


  def relabel_independent_transitions(t1, sprites1, t2, sprites2,
                                      total_samples=10, flattened=True,
                                      custom_get_mask=None, sample_multiplier=3,
                                      max_ccs=6):
    """
    Takes two transitions with their sprite representation, and combines them
    using connected-component relabeling
    """
    if flattened:
      get_mask = get_true_flat_mask
    else:
      get_mask = get_true_abstract_mask

    if custom_get_mask is not None:
      get_mask = custom_get_mask

    s1_1, a_1, _, s2_1 = t1
    s1_2, a_2, _, s2_2 = t2
    action_start = len(s1_1)

    if flattened:
      s1_1 = s1_1.flatten()
      s2_1 = s2_1.flatten()
      s1_2 = s1_2.flatten()
      s2_2 = s2_2.flatten()
      action_start = len(s1_1)
      action_idxs = list(range(action_start, action_start + len(a_1)))
    else:
      action_idxs = [action_start]

    cc1 = get_cc_from_sprites_and_action(sprites1, a_1, get_mask)
    cc1 = reduce_cc_list_by_union(cc1, max_ccs)

    cc2 = get_cc_from_sprites_and_action(sprites2, a_2, get_mask)
    cc2 = reduce_cc_list_by_union(cc2, max_ccs)

    dc1 = disentangled_components(cc1)
    dc2 = disentangled_components(cc2)
    res = []

    no_relabeling = False

    # subsample dc1 according to total_samples * sample multipler
    # sample multiplier is meant to oversample, then trim down to total_samples
    if total_samples is not None and len(dc1) > 2:
      dc1 = [()] + list(
        np.random.choice(dc1[1:], total_samples * sample_multiplier - 1))
    elif total_samples is not None:
      dc1 = dc1 * (total_samples // 2)
      no_relabeling = True

    for dc in dc1:
      # First check if disconnected component is also in the second transitions
      # Else do nothing
      if dc in dc2:
        # Given a match, we try relabeling
        proposed_sprites = copy.deepcopy(sprites1)
        proposed_action = a_1.copy()
        proposed_s1 = s1_1.copy()
        proposed_s2 = s2_1.copy()

        for idx in dc:
          if idx in action_idxs:
            if flattened:
              proposed_action[idx - action_start] = a_2[idx - action_start]
            else:
              proposed_action = a_2
          else:
            proposed_s1[idx] = s1_2[idx]
            proposed_s2[idx] = s2_2[idx]
            if flattened:
              if idx % FLAGS.num_sprites == 0:
                proposed_sprites[idx // FLAGS.num_sprites] = copy.deepcopy(
                  sprites2
                )[idx // FLAGS.num_sprites]
            else:
              proposed_sprites[idx] = copy.deepcopy(sprites2)[idx]

        # Now we also need to check if the proposal is valid
        # NOTE: This also uses custom_get_mask
        if not (get_mask in [get_random_flat_mask, get_fully_connected_mask]):
          proposed_cc = get_cc_from_sprites_and_action(proposed_sprites,
                                                       proposed_action,
                                                       get_mask)
          proposed_dc = disentangled_components(proposed_cc)
          if dc in proposed_dc:
            res.append((proposed_s1, proposed_action, 0, proposed_s2))
        else:
          res.append((proposed_s1, proposed_action, 0, proposed_s2))

    while len(res) < total_samples:
      res.append(res[np.random.choice(len(res))])

    return res[:total_samples]


  def relabel(args):
    return relabel_independent_transitions(*args)


  def enlarge_dataset(data, sprites, num_pairs, relabel_samples_per_pair,
                      flattened=True, custom_get_mask=None):
    data_len = len(data)
    all_idx_pairs = np.array(
      np.meshgrid(np.arange(data_len), np.arange(data_len))).T.reshape(-1, 2)
    chosen_idx_pairs_idxs = np.random.choice(len(all_idx_pairs), num_pairs)
    chosen_idx_pairs = all_idx_pairs[chosen_idx_pairs_idxs]

    args = []
    for (i, j) in chosen_idx_pairs:
      args.append((data[i], sprites[i], data[j], sprites[j],
                   relabel_samples_per_pair, flattened, custom_get_mask))

    with mp.Pool(min(mp.cpu_count() - 1, 16)) as pool:
      reses = pool.map(relabel, args)
    return sum(reses, [])

  res = enlarge_dataset(data, sprites, FLAGS.num_pairs,
                        FLAGS.relabel_samples_per_pair, flattened=True,
                        custom_get_mask=get_true_flat_mask)

  logging.info(' '.join(1, len(res)))
  logging.info(' '.join(2, res[:1])) # each is s1, a, r, s2

  class StateActionStateRelabeledDataset(torch.utils.data.Dataset):
    """Relabels the data up front using relabel_strategy"""

    def __init__(self, data, sprites,
                 relabel_strategy=RelabelStrategy.GROUND_TRUTH,
                 relabel_pairs=10000, samples_per_pair=5, custom_get_mask=None):
      if custom_get_mask is not None:
        get_mask = custom_get_mask
      elif relabel_strategy is RelabelStrategy.NONE:
        get_mask = get_fully_connected_mask
      elif relabel_strategy is RelabelStrategy.GROUND_TRUTH:
        get_mask = get_true_flat_mask
      elif relabel_strategy is RelabelStrategy.RANDOM:
        get_mask = get_random_flat_mask
      else:
        raise NotImplementedError

      self.data = enlarge_dataset(
        data, sprites, relabel_pairs, samples_per_pair,
        flattened=True, custom_get_mask=get_mask)

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


  tr_none = StateActionStateRelabeledDataset(data, sprites,
                                             RelabelStrategy.NONE)
  tr_true = StateActionStateRelabeledDataset(data, sprites,
                                             RelabelStrategy.GROUND_TRUTH)
  tr_rand = StateActionStateRelabeledDataset(data, sprites,
                                             RelabelStrategy.RANDOM)
  te = StateActionTestDataset(test_data, test_sprites)

  with open(os.path.join(FLAGS.results_dir, 'forward_model.pkl'), 'wb') as f:
    pickle.dump((tr_none, tr_true, tr_rand, te), f)

  logging.info(' '.join(len(tr_none), len(tr_true), len(tr_rand)))

  for tr in [tr_none, tr_true, tr_rand]:
    lst = [tr[i][0].numpy().round(2) for i in range(len(tr_none))]
    s = set([tuple(a) for a in lst])
    logging.info("Number of unique (s1, a) pairs in dataset: {}".format(len(s)))

  tr_loss_none, te_loss_none = train_fwd_model(tr_none, te)
  tr_loss_rand, te_loss_rand = train_fwd_model(tr_rand, te)
  tr_loss_true, te_loss_true = train_fwd_model(tr_true, te)

  plot_losses(tr_loss_none, te_loss_none,
              tr_loss_rand, te_loss_rand,
              tr_loss_true, te_loss_true)
