import numpy as np
import copy
from functools import reduce
from itertools import combinations, chain
from scipy.sparse.csgraph import connected_components
import multiprocessing as mp


def get_true_abstract_mask(sprites, config, action=(0.5, 0.5)):
  """Returns a mask with iteractions for next transition given true sprites.
  E.g., returns [[1,0,0],[0,1,0],[0,0,1]] for 3 sprites"""
  sprites1 = copy.deepcopy(sprites)
  config['action_space'].step(action, sprites1)
  return config['renderers']['mask_abstract'].render(sprites1)


def get_true_flat_mask(sprites, config, action=(0.5, 0.5)):
  """Returns a mask with iteractions for next transition given true sprites.
  E.g., returns [[1,0,0],[0,1,0],[0,0,1]] for 3 sprites"""
  sprites1 = copy.deepcopy(sprites)
  config['action_space'].step(action, sprites1)
  return config['renderers']['mask'].render(sprites1)


def get_random_flat_mask(sprites, config, action=(0.5, 0.5)):
  sprites1 = copy.deepcopy(sprites)
  config['action_space'].step(action, sprites1)
  mask = config['renderers']['mask'].render(sprites1)
  return np.eye(len(mask))


def get_fully_connected_mask(sprites, config, action=(0.5, 0.5)):
  sprites1 = copy.deepcopy(sprites)
  config['action_space'].step(action, sprites1)
  mask = config['renderers']['mask'].render(sprites1)
  return np.ones(mask.shape)



def powerset(n):
  xs = list(range(n))
  return list(chain.from_iterable(combinations(xs, n) for n in range(n + 1)))


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


def get_cc_from_sprites_and_action(sprites, config, action=(0.5, 0.5), get_mask=get_true_abstract_mask):
  """Returns list of connected component indices for next transition
  interactions given true sprites.

  E.g., if mask is [[1,0,0,0],[0,1,0,0],[0,0,1,1],[0,0,1,1]],
  this will return [array([0]), array([1]), array([2, 3])]
  """
  return cc_list(connected_components(get_mask(sprites, config, action)))


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
    res.append(reduce(np.union1d, [cc_lst[i] for i in subset], np.array([])).astype(np.int64))
  return list(map(tuple, res))


def relabel_independent_transitions(t1,
                                    sprites1,
                                    t2,
                                    sprites2,
                                    config,
                                    reward_fn=lambda _: 0,
                                    total_samples=10,
                                    flattened=True,
                                    custom_get_mask=None,
                                    sample_multiplier=3,
                                    max_ccs=6):
  """
  Takes two transitions with their sprite representation, and combines them
  using connected-component relabeling
  """
  num_sprites = len(sprites1)

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

  cc1 = get_cc_from_sprites_and_action(sprites1, config, a_1, get_mask)
  cc1 = reduce_cc_list_by_union(cc1, max_ccs)

  cc2 = get_cc_from_sprites_and_action(sprites2, config, a_2, get_mask)
  cc2 = reduce_cc_list_by_union(cc2, max_ccs)

  dc1 = disentangled_components(cc1)
  dc2 = disentangled_components(cc2)
  res = []

  no_relabeling = False

  # subsample dc1 according to total_samples * sample multipler
  # sample multiplier is meant to oversample, then trim down to total_samples
  if total_samples is not None and len(dc1) > 2:
    dc1 = [()] + list(np.random.choice(dc1[1:], total_samples * sample_multiplier - 1))
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
            if idx % num_sprites == 0:
              proposed_sprites[idx // num_sprites] = copy.deepcopy(sprites2)[idx // num_sprites]
          else:
            proposed_sprites[idx] = copy.deepcopy(sprites2)[idx]

      # Now we also need to check if the proposal is valid
      # NOTE: This also uses custom_get_mask
      if not (get_mask in [get_random_flat_mask, get_fully_connected_mask]):
        proposed_cc = get_cc_from_sprites_and_action(proposed_sprites, config, proposed_action, get_mask)
        proposed_dc = disentangled_components(proposed_cc)
        if dc in proposed_dc:
          res.append((proposed_s1, proposed_action, reward_fn(proposed_s2), proposed_s2))
      else:
        res.append((proposed_s1, proposed_action, reward_fn(proposed_s2), proposed_s2))

  while len(res) < total_samples:
    res.append(res[np.random.choice(len(res))])

  return res[:total_samples]


def relabel(args):
  return relabel_independent_transitions(*args)


def enlarge_dataset(data, sprites, config, num_pairs, relabel_samples_per_pair, flattened=True,
                    custom_get_mask=None, pool=True):
  data_len = len(data)
  all_idx_pairs = np.array(np.meshgrid(np.arange(data_len), np.arange(data_len))).T.reshape(-1, 2)
  chosen_idx_pairs_idxs = np.random.choice(len(all_idx_pairs), num_pairs)
  chosen_idx_pairs = all_idx_pairs[chosen_idx_pairs_idxs]
  reward_fn = config['task'].reward_of_vector_repr

  config = {
    'action_space': copy.deepcopy(config['action_space']),
    'renderers': {
      'mask': copy.deepcopy(config['renderers']['mask']),
      'mask_abstract': copy.deepcopy(config['renderers']['mask_abstract'])
    }
  }
  args = []
  for (i, j) in chosen_idx_pairs:
    args.append(
        (data[i], sprites[i], data[j], sprites[j], config, reward_fn, relabel_samples_per_pair, flattened, custom_get_mask))

  if pool:
    with mp.Pool(min(mp.cpu_count() - 1, 16)) as pool:
      reses = pool.map(relabel, args)
  else:
    reses = [relabel(_args) for _args in args]
  reses = sum(reses, [])

  return reses
