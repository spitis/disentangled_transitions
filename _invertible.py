### SEE https://github.com/pclucas14/pytorch-glow/blob/master/invertible_layers.py for more ops

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as wn
import numpy as np


# ------------------------------------------------------------------------------
# Basic MLP
# ------------------------------------------------------------------------------

def NN(input_size, layer_size=None, num_layers=2):
  if layer_size is None:
    layer_size = input_size
  layers = [nn.Linear(input_size, layer_size), nn.ReLU(inplace=True)]
  for _ in range(num_layers - 1):
    layers += [nn.Linear(layer_size, layer_size), nn.ReLU(inplace=True)]
  return nn.Sequential(*layers, nn.Linear(layer_size, layer_size))

# ------------------------------------------------------------------------------
# Abstract Classes to define common interface for invertible functions
# ------------------------------------------------------------------------------


class ReversibleLayer(nn.Module):
  """Abstract Class for bijective functions"""

  def forward(self, x):
    return self.forward_(x)[0]

  def reverse(self, x):
    return self.reverse_(x)[0]

  def forward_(self, x, objective=None):
    raise NotImplementedError

  def reverse_(self, x, objective=None):
    raise NotImplementedError

class NoDetReversibleLayer(ReversibleLayer):
  """Abstract Class for bijective functions"""

  def forward(self, x):
    raise NotImplementedError

  def reverse(self, x):
    raise NotImplementedError

  def forward_(self, x, objective=None):
    raise NotImplementedError('This layer does not implement determinant!')

  def reverse_(self, x, objective=None):
    raise NotImplementedError('This layer does not implement determinant!')

class ReversibleLayerList(ReversibleLayer):
  """Wrapper for stacking multiple layers"""

  def __init__(self, list_of_layers=None):
    super(ReversibleLayerList, self).__init__()
    self.layers = nn.ModuleList(list_of_layers)

  def __getitem__(self, i):
    return self.layers[i]

  def forward_(self, x, objective=None):
    for layer in self.layers:
      x = layer.forward(x)
    return x

  def reverse_(self, x, objective=None):
    for layer in reversed(self.layers):
      x = layer.reverse(x)
    return x

  def forward_(self, x, objective=None):
    for layer in self.layers:
      x, objective = layer.forward_(x, objective)
    return x, objective

  def reverse_(self, x, objective=None):
    for layer in reversed(self.layers):
      x, objective = layer.reverse_(x, objective)
    return x, objective

# ------------------------------------------------------------------------------
# Permutation Layers
# ------------------------------------------------------------------------------

class Shuffle(ReversibleLayer):
  """Shuffles features on dim 2 (feats or channels)"""

  def __init__(self, num_feats):
    super(Shuffle, self).__init__()

    indices = np.arange(num_feats)
    np.random.shuffle(indices)

    rev_indices = np.zeros_like(indices)
    for i in range(num_feats):
      rev_indices[indices[i]] = i

    indices = torch.from_numpy(indices).long()
    rev_indices = torch.from_numpy(rev_indices).long()
    self.register_buffer('indices', indices)
    self.register_buffer('rev_indices', rev_indices)

  def forward_(self, x, objective=None):
    return x[:, self.indices], objective

  def reverse_(self, x, objective=None):
    return x[:, self.rev_indices], objective


class Reverse(Shuffle):
  """Reverses features on dim 2 (feats or channels)"""

  def __init__(self, num_feats):
    super(Reverse, self).__init__(num_feats)
    indices = np.copy(np.arange(num_feats)[::-1])
    indices = torch.from_numpy(indices).long()
    self.indices.copy_(indices)
    self.rev_indices.copy_(indices)


class ReversibleLinear(ReversibleLayer, nn.Linear):
  """Linear fn of features"""

  def __init__(self, num_feats):
    self.num_feats = num_feats
    nn.Linear.__init__(self, num_feats, num_feats, bias=False)
 
  def reset_parameters(self):
    # initialization done with rotation matrix
    w_init = np.linalg.qr(np.random.randn(self.num_feats, self.num_feats))[0]
    w_init = torch.from_numpy(w_init.astype('float32'))
    self.weight.data.copy_(w_init)

  def forward_(self, x, objective=None):
    if objective is not None:
      dlogdet = torch.det(self.weight).abs().log()
      objective += dlogdet
    output = F.linear(x, self.weight, self.bias)

    return output, objective

  def reverse_(self, x, objective=None):
    if objective is not None:
      dlogdet = torch.det(self.weight).abs().log()
      objective -= dlogdet
    weight_inv = torch.inverse(self.weight)
    output = F.linear(x, weight_inv, self.bias)
    return output, objective

# ------------------------------------------------------------------------------
# Coupling Layers
# ------------------------------------------------------------------------------


class AdditiveCoupling(ReversibleLayer):

  def __init__(self, num_features):
    super(AdditiveCoupling, self).__init__()
    assert num_features % 2 == 0
    self.NN = NN(num_features // 2)
    self.reset_parameters()

  def reset_parameters(self):
    """init last layer to 0 so initially identity---per GLOW paper"""
    layer = self.NN[-1]
    layer.weight.data.copy_(torch.zeros_like(layer.weight.data))
    layer.bias.data.copy_(torch.zeros_like(layer.bias.data))

  def forward_(self, x, objective=None):
    z1, z2 = torch.chunk(x, 2, dim=-1)
    z2 = z2 + self.NN(z1)
    return torch.cat([z1, z2], dim=-1), objective

  def reverse_(self, x, objective=None):
    z1, z2 = torch.chunk(x, 2, dim=-1)
    z2 = z2 - self.NN(z1)
    return torch.cat([z1, z2], dim=-1), objective


class AffineCoupling(ReversibleLayer):

  def __init__(self, num_features):
    super(AffineCoupling, self).__init__()
    assert num_features % 2 == 0
    self.NN = NN(num_features // 2, num_features)
    self.reset_parameters()

  def reset_parameters(self):
    """init last layer to 0 so initially identity---per GLOW paper"""
    layer = self.NN[-1]
    layer.weight.data.copy_(torch.zeros_like(layer.weight.data))
    layer.bias.data.copy_(torch.zeros_like(layer.bias.data))


  def forward_(self, x, objective=None):
    z1, z2 = torch.chunk(x, 2, dim=-1)
    h = self.NN(z1)
    shift, h = torch.chunk(h, 2, dim=-1)
    scale = torch.tanh(h) + 1.

    z2 = z2 * scale
    z2 = z2 + shift

    if objective is not None:
      objective += torch.log(scale).sum(dim=-1)

    return torch.cat([z1, z2], dim=-1), objective

  def reverse_(self, x, objective=None):
    z1, z2 = torch.chunk(x, 2, dim=-1)
    h = self.NN(z1)
    shift, h = torch.chunk(h, 2, dim=-1)
    scale = torch.tanh(h) + 1.

    z2 = z2 - shift
    z2 = z2 / scale

    if objective is not None:
      objective -= torch.log(scale).sum(dim=-1)

    return torch.cat([z1, z2], dim=-1), objective



# ------------------------------------------------------------------------------
# Learned Autoencoder
# ------------------------------------------------------------------------------

class Flatten(nn.Module):
  def forward(self, x):
    return x.view(x.size(0), -1)

class Unflatten(nn.Module):
  def forward(self, x):
    return x.view(x.size(0), x.size(1), 1, 1)

class Crop(nn.Module):
  def __init__(self, crop=1):
    super().__init__()
    self.crop = crop
    
  def forward(self, x):
    return x[:,:,1:-1,1:-1]

class UnNormalize(object):
  def __init__(self, mean, std):
    self.mean = torch.tensor(mean).view(1, 3, 1, 1)
    self.std = torch.tensor(std).view(1, 3, 1, 1)

  def __call__(self, tensor):
    return tensor * self.std + self.mean

class ConvAutoencoder(NoDetReversibleLayer):
  def __init__(self, encoder, decoder):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
  
  def forward(self, x):
    return self.encoder(x)

  def reverse(self, x):
    return self.decoder(x)