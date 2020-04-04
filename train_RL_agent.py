"""based on td3 code: https://github.com/sfujim/TD3"""

import numpy as np
import torch
import gym
import argparse
import json
import os
import pickle
from data_utils import make_env, SpriteMaker
from functools import partial
import time

from agents import utils
from agents import TD3
from agents import DDPG

from coda import get_true_abstract_mask, get_true_flat_mask, get_random_flat_mask, get_fully_connected_mask
from coda import enlarge_dataset
from structured_transitions import MixtureOfMaskedNetworks, SimpleStackedAttn

""" MULTIPROCESSING STUFF """
import torch.multiprocessing as mp
if __name__ == '__main__':
  mp.set_start_method('spawn')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL = None


def model_get_mask(sprites, config, action, MODEL, THRESH):
  del config  # unused
  global DEVICE

  state = np.vstack(
    [np.hstack((sprite.x, sprite.y, sprite.velocity))
      for sprite in sprites]
  )  # N_sprites x 4 matrix
  state = torch.FloatTensor(state.ravel())
  action = torch.FloatTensor(action)
  state = state.unsqueeze(0)  # expand batch dim
  action = action.unsqueeze(0)  # expand batch dim
  model_input = torch.cat((state, action), -1)
  model_input = model_input.to(DEVICE)

  with torch.no_grad():
    _, mask, _ = MODEL.forward_with_mask(model_input)
    # add dummy columns for (state, action -> next action) portion
    mask = mask.squeeze().cpu()
    dummy_columns = torch.zeros(len(mask), 2)
    mask = torch.cat((mask, dummy_columns), -1)
  mask = mask.cpu().numpy()
  mask = (mask > THRESH).astype(np.float32)
  return mask

""" END MULTIPROCESSING STUFF """


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, seed, num_sprites, eval_episodes=50, episode_len=100,
                reward_type='min_pairwise'):
  _, eval_env = make_env(num_sprites=num_sprites, reward_type=reward_type)
  eval_env = SymmetricActionWrapper(FlatEnvWrapper(eval_env))  
  eval_env.seed(seed + 100)

  avg_reward = 0.
  for _ in range(eval_episodes):
    state, done = eval_env.reset(), False
    steps = 0
    while not done:
      action = policy.select_action(np.array(state))
      state, reward, done, _ = eval_env.step(action)
      steps += 1
      if steps >= episode_len:
        done=True
      avg_reward += reward
      
  avg_reward /= (eval_episodes * episode_len)
  return avg_reward


class FlatEnvWrapper(gym.ObservationWrapper):
  """Flattens the environment observations so that only the disentangled observation is returned."""
  def observation(self, observation):
    return observation['disentangled'].flatten()


class SymmetricActionWrapper(gym.ActionWrapper):
  """Turns transforms action from (-1, 1) to (0, 1)."""
  def action(self, action):
    return (action + 1.) / 2.


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--num_sprites', type=int, default=4, help='Number of sprites.')
  # TODO(creager): make work with variable num_sprites
  parser.add_argument("--policy", default="TD3")  # Policy name (TD3, DDPG or OurDDPG)
  parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
  parser.add_argument("--start_timesteps", default=5e3, type=int)  # Time steps initial random policy is used
  parser.add_argument("--eval_freq", default=1e3, type=int)  # How often (time steps) we evaluate
  parser.add_argument("--max_timesteps", default=5e5, type=int)  # Max time steps to run environment
  parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
  parser.add_argument("--batch_size", default=1000, type=int)  # Batch size for both actor and critic
  parser.add_argument("--discount", default=0.99)  # Discount factor
  parser.add_argument("--tau", default=0.005)  # Target network update rate
  parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
  parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
  parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
  parser.add_argument("--episode_len", default=50, type=int)  # Episode length before reset
  parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
  parser.add_argument("--save_replay", action="store_true")  # Save replay buffer
  parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
  parser.add_argument("--reward_type", default="min_pairwise")  # Which reward function to use
  parser.add_argument("--relabel_type", default=None, type=str)  # type of relabeling to do
  parser.add_argument("--relabel_every", default=1000, type=int)  # how often to do relabeling
  parser.add_argument('--num_pairs', type=int, default=2000, help='Number of transition pairs to sample for relabeling.')
  parser.add_argument('--attn_mech_dir', type=str, default='/tmp/spriteworld_scm_discovery', help='Path to folder containing trained attention mechanism model and its kwargs.')
  parser.add_argument('--coda_samples_per_pair', type=int, default=5, help='Number of relabels per transition pairs.')
  parser.add_argument('--opt_steps_per_env_step', type=int, default=1)
  parser.add_argument('--tag', type=str, default='')
  parser.add_argument('--results_dir', type=str, default='.')
  parser.add_argument("--thresh", default=0.1, type=float, help='Threshold on attention mask')
  parser.add_argument("--max_cpu", default=8, type=int, help='CPUs to use')

  args = parser.parse_args()

  file_name = f"{args.policy}_{args.reward_type}_{args.relabel_type}_{args.tag}__{args.seed}"
  print("---------------------------------------")
  print(f"Policy: {args.policy}, Env: Bouncing Balls, Seed: {args.seed}")
  print("---------------------------------------")

  if not os.path.exists(os.path.join(args.results_dir, 'results')):
    os.makedirs(os.path.join(args.results_dir, 'results'))

  if args.save_model and not os.path.exists(os.path.join(args.results_dir,
                                                         'models')):
    os.makedirs(os.path.join(args.results_dir, 'models'))

  config, original_env = make_env(
    num_sprites=args.num_sprites, reward_type=args.reward_type)
  _, env = make_env(num_sprites=args.num_sprites, reward_type=args.reward_type)
  env = SymmetricActionWrapper(FlatEnvWrapper(env))
  state_to_sprites = SpriteMaker()

  # Set seeds
  env.seed(args.seed)
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  
  state_dim = env.observation_space['disentangled'].shape[0]
  action_dim = env.action_space.shape[0]
  max_action = float(env.action_space.high[0])

  kwargs = {
    "state_dim": state_dim,
    "action_dim": action_dim,
    "max_action": max_action,
    "discount": args.discount,
    "tau": args.tau,
  }

  # Initialize policy
  if args.policy == "TD3":
    # Target policy smoothing is scaled wrt the action scale
    kwargs["policy_noise"] = args.policy_noise * max_action
    kwargs["noise_clip"] = args.noise_clip * max_action
    kwargs["policy_freq"] = args.policy_freq
    policy = TD3.TD3(**kwargs)
  elif args.policy == "DDPG":
    policy = DDPG.DDPG(**kwargs)

  if args.load_model != "":
    policy_file = file_name if args.load_model == "default" else args.load_model
    policy.load(f"./models/{policy_file}")

  replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
  coda_buffer = utils.ReplayBuffer(state_dim, action_dim, max_size=int(4e6))

  eval_episodes = 50
  # Evaluate untrained policy
  evaluations = [eval_policy(policy, args.seed, args.num_sprites, eval_episodes=eval_episodes, reward_type=args.reward_type)]
  TIME = time.time()
  print(f"Time 0 -- Evaluation over {eval_episodes} episodes: {evaluations[-1]:.3f} --- coda_buffer length: {len(coda_buffer)}")

  episode_reward = 0
  episode_timesteps = 0
  episode_num = 0

  # format attention mechanism (learned or ground truth) as callable mask fn
  if args.relabel_type:
    if args.relabel_type == 'ground_truth':
      get_mask = get_true_flat_mask
    elif args.relabel_type == 'attn_mech':
      model_path = os.path.join(args.attn_mech_dir, 'model.p')
      model_kwargs_path = os.path.join(args.attn_mech_dir,
                                       'model_kwargs.json')
      with open(model_kwargs_path) as f:
        model_kwargs = json.load(f)
      device = 'cuda' if torch.cuda.is_available() else 'cpu'
      #device = 'cpu'

      # A hack for now... 
      if model_kwargs['in_features'] == 4:
        attn_mech = SimpleStackedAttn(**model_kwargs).to(device)
      else:
        attn_mech = MixtureOfMaskedNetworks(**model_kwargs).to(device)

      attn_mech.load_state_dict(torch.load(model_path))
      attn_mech.to(device)
      attn_mech.eval()

      MODEL = attn_mech

      get_mask = partial(model_get_mask, MODEL=MODEL, THRESH=args.thresh)
    else:
      raise NotImplementedError

  # train agent
  for t in range(int(args.max_timesteps)):
    
    if t % args.episode_len == 0:
      state, done = env.reset(), False

    episode_timesteps += 1

    # Select action randomly or according to policy
    if t < args.start_timesteps:
      action = env.action_space.sample()
    else:
      action = (
        policy.select_action(np.array(state))
        + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
      ).clip(-max_action, max_action)

    # Perform action
    next_state, reward, done, _ = env.step(action)
    done_bool = float(done) if episode_timesteps < args.episode_len else 0

    # Store data in replay buffer
    replay_buffer.add(state, action, next_state, reward, done_bool)

    # Do coda if enabled
    if (args.relabel_type) and (len(replay_buffer) % args.relabel_every) == 0:
      base_data = replay_buffer.sample_list_of_sars(args.num_pairs) #reusing numpairs here...
      sprites_for_base_data = [state_to_sprites(state) for state, _, _, _ in base_data]

      lst = [state.round(2) for state, _, _, _ in base_data]
      og_state_set = set([tuple(a) for a in lst])


      pool = True #args.relabel_type != 'attn_mech'  # attn mechanism doesn't pool
      coda_data = enlarge_dataset(base_data,
                                  sprites_for_base_data,
                                  config,
                                  args.num_pairs,
                                  args.coda_samples_per_pair,
                                  flattened=True,
                                  custom_get_mask=get_mask,
                                  pool=pool,
                                  max_cpus = 15 if not args.relabel_type == 'attn_mech' else args.max_cpu)

      # remove duplicates of original data
      coda_data = [(s, a, r, s2) for s, a, r, s2 in coda_data if not tuple(s.round(2)) in og_state_set]

      for (s, a, r, s2) in coda_data:
        coda_buffer.add(s, a, s2, r, 0) # note weird add order. 

    state = next_state
    episode_reward += reward

    # Train agent after collecting sufficient data
    if t >= args.start_timesteps:
      for _ in range(args.opt_steps_per_env_step):
        real_frac = float(len(replay_buffer)) / (len(replay_buffer) + len(coda_buffer))
        real_batch_size = int(real_frac * args.batch_size)
        coda_batch_size = args.batch_size - real_batch_size

        samples = replay_buffer.sample(real_batch_size)
        if coda_batch_size:
          coda_samples = coda_buffer.sample(coda_batch_size)
          samples = [torch.cat((a, b), 0) for a, b in zip(samples, coda_samples)]

        policy.train(samples)

    if done:
      raise ValueError("WTIH SPRITEWORLD WE SHOULD NEVER SEE DONE")
      # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
      print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
      # Reset environment
      state, done = env.reset(), False
      episode_reward = 0
      episode_timesteps = 0
      episode_num += 1 

    # Evaluate episode
    if (t + 1) % args.eval_freq == 0:
      evaluations.append(eval_policy(policy, args.seed, args.num_sprites, eval_episodes=eval_episodes, reward_type=args.reward_type))
      print(f"Time {t+1} -- Evaluation over {eval_episodes} episodes: {evaluations[-1]:.3f} --- coda_buffer length: {len(coda_buffer)}  ---- time: {time.time() - TIME}s")
      TIME = time.time()
      np.save(f"{args.results_dir}/results/{file_name}", evaluations)
      if args.save_model: 
        policy.save(f"{args.results_dir}/models/{file_name}")
      if args.save_replay:
        with open(f"{args.results_dir}/models/{file_name}_replay.pickle", 'wb') as f:
          pickle.dump(replay_buffer, f)