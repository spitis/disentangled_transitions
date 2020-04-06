"""Load attn model and make movie of its rollouts in an environment."""
import argparse
import json
import logging
import os
import sys

import torch

from data_utils import make_env
from plot_utils import anim_with_attn
from structured_transitions import MixtureOfMaskedNetworks, SimpleStackedAttn


if __name__ == "__main__":
  parser = argparse.ArgumentParser("Sample model-based rollouts.")
  parser.add_argument('--results_dir',
                      type=str,
                      default=None,
                      help='Output directory.')
  parser.add_argument('--num_sprites',
                      type=int,
                      default=4, help='Number of sprites.')
  parser.add_argument('--imagedim', 
                      type=int, 
                      default=16, 
                      help='Image dimension.')
  parser.add_argument('--num_examples',
                      type=int,
                      default=500,
                      help='Number of examples in dynamics model training set.')
  parser.add_argument('--num_frames',
                      type=int,
                      default=200,
                      help='Number of frames in sampled rollouts videos.')
  parser.add_argument('--seed',
                      type=int,
                      default=1,
                      help='Random seed.')
  parser.add_argument('--max_episode_length',
                      type=int,
                      default=5000,
                      help='Max length of an episode.')
  parser.add_argument('--model_type',
                      type=str,
                      default='MME',
                      help='Type of dynamics model.')
  parser.add_argument('--attn_mech_dir',
                      type=str,
                      default='/tmp/spriteworld_scm_discovery',
                      help='Path to folder containing trained attention '
                           'mechanism model and its kwargs.')
  parser.add_argument('--thresh',
                      type=float,
                      default=0.05,
                      help='Threshold on attention mask.')
  parser.add_argument('--fps',
                      type=float,
                      default=6,
                      help='Frames per second of video.')
  FLAGS = parser.parse_args()

  # if no results dir specified, write movie to a subdir of the attn mech dir
  FLAGS.results_dir = FLAGS.results_dir or os.path.join(
    FLAGS.attn_mech_dir, 'rollouts_with_attn'
  )

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
  log_filename = os.path.join(FLAGS.results_dir, 'main.log')
  if os.path.exists(log_filename):
    os.remove(log_filename)
  logging.basicConfig(filename=log_filename, level=logging.INFO)
  # log to std err
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  logging.getLogger('').addHandler(console)

  ground_truth_kwargs = dict(num_sprites=FLAGS.num_sprites, seed=FLAGS.seed,
    max_episode_length=FLAGS.max_episode_length, imagedim=FLAGS.imagedim)
  config, env = make_env(**ground_truth_kwargs)
  env.action_space.seed(FLAGS.seed)  # reproduce randomness in action space

  # load attn mech from disk
  model_path = os.path.join(FLAGS.attn_mech_dir, 'model.p')
  model_kwargs_path = os.path.join(FLAGS.attn_mech_dir,
                                   'model_kwargs.json')
  with open(model_kwargs_path) as f:
    model_kwargs = json.load(f)
  # device = 'cuda' if torch.cuda.is_available() else 'cpu'
  device = 'cpu'
  if FLAGS.model_type == 'MME':
    attn_mech = MixtureOfMaskedNetworks(**model_kwargs)
  elif FLAGS.model_type == 'SSA':
    attn_mech = SimpleStackedAttn(**model_kwargs)
  else:
    raise NotImplementedError
  
  attn_mech.load_state_dict(torch.load(model_path))
  attn_mech.to(device)
  attn_mech.eval()

  # write movie of actual environment rollouts
  plot_kwargs = dict(show_resets=True, show_clicks=True, fps=FLAGS.fps)
  basename = 'rollouts_with_attn_thresh_{}.mp4'.format(FLAGS.thresh)
  res = anim_with_attn(env,
                       attn_mech,
                       FLAGS.thresh,
                       FLAGS.num_frames,
                       filename=os.path.join(FLAGS.results_dir, basename),
                       **plot_kwargs)
  logging.info('done')

