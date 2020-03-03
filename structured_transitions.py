import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


class GELU(nn.Module):
  def forward(self, input):
    return F.gelu(input)


class Randfn(nn.Module):
  """
  A random function from R^in --> R^out. 
  These are composed to make a transition function.
  """
  def __init__(self, num_in, num_out):
    super().__init__()
    self.f = nn.Sequential(*[nn.Linear(num_in, 32), GELU(), nn.Linear(32, num_out)])

  def forward(self, x):
    return self.f(x)


class MaskedLinear(nn.Linear):
  """ same as Linear except returns absolute weights """
  @property
  def mask(self):
    return torch.abs(self.weight).T


class MaskedNetwork(nn.Module):
  """ same as MLP except has a mask on the weights """
  def __init__(self, in_features, out_features, num_hidden_layers=2, num_hidden_units=256):
    super().__init__()
    layers = [MaskedLinear(in_features, num_hidden_units), nn.ReLU()]
    for _ in range(num_hidden_layers - 1):
      layers += [MaskedLinear(num_hidden_units, num_hidden_units), nn.ReLU()]
    layers += [MaskedLinear(num_hidden_units, out_features)]
    self._in = in_features
    self.f = nn.Sequential(*layers)

  @property
  def mask(self):
    res = self.f[0].mask

    for layer in self.f[1:]:
      if hasattr(layer, 'mask'):
        res = res.matmul(layer.mask)

    return res

  def forward(self, x):
    return self.f(x)


class MixtureOfMaskedNetworks(nn.Module):
  def __init__(self, in_features, out_features, num_components=10, num_hidden_layers=2, num_hidden_units=256):
    super().__init__()
    self.components = nn.ModuleList(
        [MaskedNetwork(in_features, out_features, num_hidden_layers, num_hidden_units) for _ in range(num_components)])

    self.attn_net = nn.Sequential(
        *[nn.Linear(in_features, num_hidden_units),
          nn.ReLU(inplace=True),
          nn.Linear(num_hidden_units, num_components)])

  def dynamic_mask(self, x):
    return self.forward_with_mask(x)[1]

  def forward(self, x):
    return self.forward_with_mask(x)[0]

  def forward_with_mask(self, x):
    masks = torch.stack([c.mask for c in self.components])[None]  # 1 x C x I x O
    comps = torch.stack([c(x) for c in self.components]).transpose(0, 1)  # B x C x O

    attns = F.softmax(self.attn_net(x), -1)  # B x C

    mask = (masks * attns[:, :, None, None]).sum(dim=1)  # B x I x O
    res = (comps * attns[:, :, None]).sum(dim=1)  # B x O

    return res, mask, attns


def gen_samples_static(num_seqs=16, seq_len=12, splits=[5, 3, 2]):
  """
  Generates num_seqs sequences of length seq_len, which are factorized according to splits,
  and returns a dataset of with total num_seqs*seq_len transitions.
  The factorization here is static, and applies to all transitions.
  """
  seq = [torch.randn(num_seqs, sum(splits))]
  fns = []

  for split in splits:
    fns.append(Randfn(split, split))

  for i in range(seq_len):
    factors = seq[-1].split(splits, dim=1)
    next_factors = [fn(f) for f, fn in zip(factors, fns)]
    seq.append(torch.cat(next_factors, 1))

  return fns, (torch.cat(seq[:-1]), torch.cat(seq[1:]))


def gen_samples_dynamic(num_seqs=16, seq_len=12, splits=[5, 3, 2], epsilon=2.):
  """
  Generates num_seqs sequences of length seq_len, which are factorized according to splits,
  and returns a dataset of with total num_seqs*seq_len transitions.
  The factorization here is dynamic, where sometimes there is a global interaction, according to epsilon. 
  Lower epsilon = MORE global interaction. 
  Note that the global interactions are all global... so there is no sparse object-to-object interaction.
  """
  seq = [torch.randn(num_seqs, sum(splits))]
  fns = []
  global_fns = []
  total_global_interactions = 0

  for split in splits:
    fns.append(Randfn(split, split))
    global_fns.append(Randfn(split, sum(splits)))

  for i in range(seq_len):
    factors = seq[-1].split(splits, dim=1)
    next_factors = [fn(f / f.norm(dim=-1).view(-1, 1)) for f, fn in zip(factors, fns)]
    seq.append(torch.cat(next_factors, 1))

    for f, gfn in zip(factors, global_fns):
      mask = (f.norm(dim=-1) > epsilon).to(torch.float32).view(-1, 1)
      total_global_interactions += mask.sum()
      seq[-1] = seq[-1] + gfn(f / f.norm(dim=-1).view(-1, 1)) * mask

  return total_global_interactions, fns, (torch.cat(seq[:-1]), torch.cat(seq[1:]))


class TransitionsData(torch.utils.data.Dataset):
  def __init__(self, samples):
    x, y = samples
    self.x = x.detach()
    self.y = y.detach()

  def __len__(self):
    return len(self.x)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    x = self.x[idx]
    y = self.y[idx]
    return x, y