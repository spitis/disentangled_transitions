"""based on td3 code: https://github.com/sfujim/TD3"""

import numpy as np
import torch
import gym
import argparse
import os
from data_utils import make_env

from agents import utils
from agents import TD3
from agents import DDPG


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(t, policy, seed, eval_episodes=50, episode_len=100, reward_type='min_pairwise'):
  _, eval_env = make_env(reward_type=reward_type)
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

  print(f"Time {t} -- Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
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
  parser.add_argument("--policy", default="TD3")  # Policy name (TD3, DDPG or OurDDPG)
  parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
  parser.add_argument("--start_timesteps", default=5e3, type=int)  # Time steps initial random policy is used
  parser.add_argument("--eval_freq", default=1e3, type=int)  # How often (time steps) we evaluate
  parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
  parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
  parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
  parser.add_argument("--discount", default=0.99)  # Discount factor
  parser.add_argument("--tau", default=0.005)  # Target network update rate
  parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
  parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
  parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
  parser.add_argument("--episode_len", default=50, type=int)  # Episode length before reset
  parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
  parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
  parser.add_argument("--reward_type", default="min_pairwise")  # Model load file name, "" doesn't load, "default" uses file_name
  args = parser.parse_args()

  file_name = f"{args.policy}_{args.seed}"
  print("---------------------------------------")
  print(f"Policy: {args.policy}, Env: Bouncing Balls, Seed: {args.seed}")
  print("---------------------------------------")

  if not os.path.exists("./results"):
    os.makedirs("./results")

  if args.save_model and not os.path.exists("./models"):
    os.makedirs("./models")
  
  config, original_env = make_env(reward_type=args.reward_type)
  _, env = make_env(reward_type=args.reward_type)
  env = SymmetricActionWrapper(FlatEnvWrapper(env))

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
  elif args.policy == "OurDDPG":
    policy = OurDDPG.DDPG(**kwargs)
  elif args.policy == "DDPG":
    policy = DDPG.DDPG(**kwargs)

  if args.load_model != "":
    policy_file = file_name if args.load_model == "default" else args.load_model
    policy.load(f"./models/{policy_file}")

  replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
  
  # Evaluate untrained policy
  evaluations = [eval_policy(0, policy, args.seed, reward_type=args.reward_type)]

  episode_reward = 0
  episode_timesteps = 0
  episode_num = 0

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

    state = next_state
    episode_reward += reward

    # Train agent after collecting sufficient data
    if t >= args.start_timesteps:
      policy.train(replay_buffer, args.batch_size)

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
      evaluations.append(eval_policy(t+1, policy, args.seed, reward_type=args.reward_type))
      np.save(f"./results/{file_name}", evaluations)
      if args.save_model: policy.save(f"./models/{file_name}")