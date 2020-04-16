import numpy as np
import scipy
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

class SimpleMLP(nn.Module):
  def __init__(self, layers):
    super().__init__()
    layer_list = []
    for i, o in zip(layers[:-1], layers[1:]):
      layer_list += [nn.Linear(i, o), nn.ReLU()]
    layer_list = layer_list[:-1]
    self.f = nn.Sequential(*layer_list)
    
  def forward(self, x):
    return self.f(x)
  
class SimpleAttn(nn.Module):
  def __init__(self, embed_layers, attn_layers):
    super().__init__()
    assert(embed_layers[0] == attn_layers[0])
    
    attn_layers = tuple(attn_layers[:-1]) + (2*attn_layers[-1],)
    
    self.embed = SimpleMLP(embed_layers)
    self.KQ = SimpleMLP(attn_layers)
    
  def forward(self, x):
    return self.forward_with_mask(x)[0]
  
  def forward_with_mask(self, x):
    embs = self.embed(x)
    K, Q = torch.chunk(self.KQ(x), chunks=2, dim=2)
    A = F.softmax(Q.bmm(K.transpose(1,2)), 2)
    
    output = A.bmm(embs) # + embs
    mask = A # + torch.eye(A.size(2), device=A.device)
    return output, mask
  
class SimpleStackedAttn(nn.Module):
  def __init__(self, in_features, out_features, num_components=2, num_hidden_layers=2, num_hidden_units=256, action_dim=2):
    """
    Arg names chosen to match MixtureOfMaskNetworks, so this can be dropped in
    """
    num_blocks = num_components

    super().__init__()
    blocks = [SimpleAttn(
      (in_features + action_dim,) + (num_hidden_units,)*num_hidden_layers, (in_features + action_dim,) + (num_hidden_units,)*num_hidden_layers)]
    for block in range(num_blocks-1):
      blocks.append(SimpleAttn((num_hidden_units,)*(1 + num_hidden_layers),(num_hidden_units,)*(1 + num_hidden_layers)))
    output_projection = nn.Linear(num_hidden_units, out_features)
    self.f = nn.Sequential(*blocks, output_projection)

    self.in_features = in_features
    self.action_dim = action_dim
    self.num_state_slots = None
    
  def forward(self, x):
    return self.f(x)
  
  def forward_with_mask(self, x):
    """x is assumed to be a state + action, where state = (slot,)*num_slots, and action_dim < slot_dim"""

    # Automatically reshape x into slots
    x_dim = x.shape[-1]

    num_slots = (x_dim - self.action_dim) // self.in_features
    
    state_feats, action_feats = torch.split(x, [x_dim - self.action_dim, self.action_dim], dim=1)
    
    state_feats = state_feats.reshape(x.shape[0], num_slots, self.in_features)
    state_feats = torch.cat( (state_feats, torch.zeros((x.shape[0], num_slots, self.action_dim), device=x.device) ), 2)
    
    action_feats = action_feats.reshape(x.shape[0], 1, self.action_dim)
    action_feats = torch.cat( (action_feats, torch.zeros((x.shape[0], 1, self.in_features), device=x.device) ), 2)
    
    x = torch.cat( (state_feats, action_feats), 1) # batch_size, num_slots + 1, in_features + action_features

    #x = torch.cat((  x, torch.zeros((x.shape[0], self.in_features - self.action_dim), device=x.device)  ), 1)
    #x = x.reshape(x.shape[0], num_slots + 1, self.in_features + self.action_dim)

    mask = torch.eye(x.size(1), device=x.device)[None].repeat(x.size(0), 1, 1)
    # masks = [mask]
    for module in self.f:
      if type(module) is SimpleAttn:
        x, m = module.forward_with_mask(x)
        # masks.append(m)
        mask = mask.bmm(m.transpose(1,2))
      else:
        x = module(x)

    # result is shaped by slot, so flatten it
    x = x[:, :num_slots, :]
    x = x.reshape(x.shape[0], -1)

    # expand the mask
    mask = mask[:, :-1, :]
    mask = mask.repeat_interleave(self.in_features, 1).repeat_interleave(self.in_features, 2)
    mask = mask[:, :, :x_dim].transpose(1,2)

    return x, mask, torch.zeros_like(x) # The zeros is because the other MixtureOfMaskedNetwork returns attns

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

  # Build ground truth masks for each transition in the dataset (all the same).
  ground_truth_sparsity = scipy.linalg.block_diag(*(
    np.ones((split, split), dtype=np.int_) for split in splits
  ))
  ground_truth_sparsity = torch.tensor(ground_truth_sparsity)
  sparsity_per_transition = ground_truth_sparsity.repeat(
    num_seqs * seq_len, 1, 1)

  transition_start = torch.cat(seq[:-1])
  transition_end = torch.cat(seq[1:])

  samples = (transition_start, transition_end, sparsity_per_transition)

  return fns, samples


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

  # the "baseline" is no global connectivity
  baseline_ground_truth_sparsity = scipy.linalg.block_diag(*(
    np.ones((split, split), dtype=np.int_) for split in splits
  ))
  baseline_ground_truth_sparsity = torch.tensor(baseline_ground_truth_sparsity)
  # local_sparsity = [baseline_ground_truth_sparsity.repeat(num_seqs, 1, 1)]
  local_sparsity = []
  fan_outs = []
  for start, end in zip(
    np.cumsum(splits) - splits[0],
    np.cumsum(splits)
  ):
    fan_out = np.zeros((sum(splits), sum(splits)))
    fan_out[:, start:end] = 1
    fan_out = torch.tensor(fan_out, dtype=torch.float32)
    fan_outs.append(fan_out)
  batched_fan_outs = [
    fo.repeat(num_seqs, 1, 1) for fo in fan_outs
  ]

  for i in range(seq_len):
    factors = seq[-1].split(splits, dim=1)
    next_factors = [fn(f / f.norm(dim=-1).view(-1, 1)) for f, fn in zip(factors, fns)]
    seq.append(torch.cat(next_factors, 1))
    local_sparsity.append(
      baseline_ground_truth_sparsity.repeat(num_seqs, 1, 1)
    )

    for f, gfn, bfo in zip(factors, global_fns, batched_fan_outs):
      mask = (f.norm(dim=-1) > epsilon).to(torch.float32).view(-1, 1)
      total_global_interactions += mask.sum()
      seq[-1] = seq[-1] + gfn(f / f.norm(dim=-1).view(-1, 1)) * mask
      # add local dynamics
      local_sparsity[-1] = local_sparsity[-1] + bfo * mask.view(-1, 1, 1)
    # binarize local dynamics
    local_sparsity[-1] = torch.where(local_sparsity[-1] > 0,
                                     torch.ones_like(local_sparsity[-1]),
                                     torch.zeros_like(local_sparsity[-1])
                                     )

  samples = (torch.cat(seq[:-1]), torch.cat(seq[1:]), torch.cat(local_sparsity))
  return total_global_interactions, fns, samples


class TransitionsData(torch.utils.data.Dataset):
  def __init__(self, samples):
    x, y, m = samples
    self.x = x.detach()
    self.y = y.detach()
    self.m = m.detach()

  def __len__(self):
    return len(self.x)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    x = self.x[idx]
    y = self.y[idx]
    m = self.m[idx]
    return x, y, m