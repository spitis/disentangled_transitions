import argparse
from enum import Enum
import logging
import os
import pickle
import sys

import numpy as np
import torch

from data_utils import make_env
from data_utils import create_factorized_dataset
from structured_transitions import MaskedNetwork
from plot_utils import anim

from coda import get_true_abstract_mask, get_true_flat_mask, get_random_flat_mask, get_fully_connected_mask
from coda import enlarge_dataset


def plot_losses(tr_loss_none, te_loss_none, tr_loss_rand, te_loss_rand, tr_loss_true, te_loss_true, results_dir):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  import seaborn as sns
  sns.set(style='white')
  plt.figure(figsize=(8, 6))
  plt.plot(te_loss_none, linestyle='-', c='gray', label='No relabeling')
  plt.plot(te_loss_rand, linestyle='-', c='blue', label='Random relabeling')
  plt.plot(te_loss_true, linestyle='-', c='orange', label='Ground truth relabeling')

  plt.ylim(0.0003, 0.0015)
  plt.plot(tr_loss_none, linestyle='--', c='gray')
  plt.plot(tr_loss_rand, linestyle='--', c='blue')
  plt.plot(tr_loss_true, linestyle='--', c='orange')

  plt.legend()
  plt.ylabel('Mean squared error', size=16)
  plt.xlabel('Epochs', size=16)
  plt.title('Learning curves for different relabeling strategies', size=16)
  plt.show
  plt.savefig(os.path.join(results_dir, 'losses.pdf'))


class RelabelStrategy(Enum):
  NONE = 0
  GROUND_TRUTH = 1
  RANDOM = 2


def train_fwd_model(tr, te, FLAGS):
  dev = 'cuda' if torch.cuda.is_available() else 'cpu'
  train_loader = torch.utils.data.DataLoader(tr,
                                             batch_size=FLAGS.batch_size,
                                             shuffle=True,
                                             num_workers=2,
                                             drop_last=True)
  test_loader = torch.utils.data.DataLoader(te,
                                            batch_size=FLAGS.batch_size,
                                            shuffle=False,
                                            num_workers=2,
                                            drop_last=True)

  model = MaskedNetwork(in_features=FLAGS.num_sprites * 4 + 2,
                        out_features=FLAGS.num_sprites * 4,
                        num_hidden_layers=2,
                        num_hidden_units=256).to(dev)
  opt = torch.optim.Adam(model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)
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
      logging.info('Epoch {} done! Train loss: {:.5f}.  Test loss: {:.5f}'.format(epoch, total_loss / i, test_loss / j))

  return tr_losses, te_losses

class StateActionStateRelabeledDataset(torch.utils.data.Dataset):
  """Relabels the data up front using relabel_strategy"""
  def __init__(self,
                data,
                sprites,
                config,
                relabel_strategy=RelabelStrategy.GROUND_TRUTH,
                relabel_pairs=10000,
                samples_per_pair=5,
                custom_get_mask=None):
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

    self.data = enlarge_dataset(data,
                                sprites,
                                config,
                                relabel_pairs,
                                samples_per_pair,
                                flattened=True,
                                custom_get_mask=get_mask)

    self.s1, self.a, _, self.s2 = list(zip(*self.data))

    self.s1 = torch.FloatTensor(self.s1).detach()
    self.a = torch.FloatTensor(self.a).detach()
    self.s2 = torch.FloatTensor(self.s2).detach()

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
  def __init__(self, data):
    self.s1, self.a, _, self.s2 = list(zip(*data))
    self.s1 = torch.FloatTensor(self.s1).detach().flatten(start_dim=1)
    self.a = torch.FloatTensor(self.a).detach()
    self.s2 = torch.FloatTensor(self.s2).detach().flatten(start_dim=1)

  def __len__(self):
    return len(self.s1)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    s1 = self.s1[idx]
    a = self.a[idx]
    s2 = self.s2[idx]
    return torch.cat((s1, a), 0), s2


def main(FLAGS):
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
  logging.basicConfig(filename=os.path.join(FLAGS.results_dir, 'main.log'), level=logging.INFO)
  # log to std err
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  logging.getLogger('').addHandler(console)

  # set random seed
  np.random.seed(FLAGS.seed)

  config, env = make_env(num_sprites=FLAGS.num_sprites,
                         seed=FLAGS.seed,
                         imagedim=FLAGS.imagedim,
                         max_episode_length=FLAGS.max_episode_length)

  res = anim(env, 200, filename=os.path.join(FLAGS.results_dir, 'env.mp4'))

  data, sprites = create_factorized_dataset(env, 2000)
  test_data, test_sprites = create_factorized_dataset(env, 5000)
  res = enlarge_dataset(data,
                        sprites,
                        config,
                        FLAGS.num_pairs,
                        FLAGS.relabel_samples_per_pair,
                        flattened=True,
                        custom_get_mask=get_true_flat_mask)

  logging.info(len(res))
  logging.info(res[:1])  # each is s1, a, r, s2

  tr_none = StateActionStateRelabeledDataset(data, sprites, config, RelabelStrategy.NONE)
  tr_true = StateActionStateRelabeledDataset(data, sprites, config, RelabelStrategy.GROUND_TRUTH)
  tr_rand = StateActionStateRelabeledDataset(data, sprites, config, RelabelStrategy.RANDOM)
  te = StateActionTestDataset(test_data)

  with open(os.path.join(FLAGS.results_dir, 'forward_model.pkl'), 'wb') as f:
    pickle.dump((tr_none, tr_true, tr_rand, te), f)

  logging.info('%d, %d, %d' % (len(tr_none), len(tr_true), len(tr_rand)))

  for name, tr in zip(('none', 'true', 'rand'), (tr_none, tr_true, tr_rand)):
    lst = [tr[i][0].numpy().round(2) for i in range(len(tr_none))]
    s = set([tuple(a) for a in lst])
    logging.info("Number of unique (s1, a) pairs in dataset {}: {}".format(name, len(s)))

  logging.info('training mask network from data with no relabeling')
  tr_loss_none, te_loss_none = train_fwd_model(tr_none, te, FLAGS)
  logging.info('training mask network from data with random relabeling')
  tr_loss_rand, te_loss_rand = train_fwd_model(tr_rand, te, FLAGS)
  logging.info('training mask network from data with ground truth relabeling')
  tr_loss_true, te_loss_true = train_fwd_model(tr_true, te, FLAGS)

  plot_losses(tr_loss_none, te_loss_none, tr_loss_rand, te_loss_rand, tr_loss_true, te_loss_true, FLAGS.results_dir)


if __name__ == "__main__":
  parser = argparse.ArgumentParser("CoDA generate data.")
  parser.add_argument('--num_sprites', type=int, default=4, help='Number of sprites.')
  parser.add_argument('--imagedim', type=int, default=16, help='Image dimension.')
  parser.add_argument('--batch_size', type=int, default=500, help='Batch size.')
  parser.add_argument('--lr', type=float, default=1e-3, help='Learining rate.')
  parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay.')
  parser.add_argument('--num_pairs', type=int, default=500, help='Number of transition pairs.')
  parser.add_argument('--relabel_samples_per_pair',
                      type=int,
                      default=10,
                      help='Number of relabels per transition pairs.')
  parser.add_argument('--seed', type=int, default=1, help='Random seed.')
  parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs of training.')
  parser.add_argument('--max_episode_length', type=int, default=5000, help='Max length of an episode.')
  parser.add_argument('--results_dir', type=str, default='/tmp/coda_forward_model', help='Output directory.')

  args = parser.parse_args()
  if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)

  main(args)