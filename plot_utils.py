from typing import Iterable

from colour import Color
import numpy as np
import torch


FPS = 12  # frames per second


def viz(obs, filename='./viz.pdf'):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt

  plt.figure(figsize=(2, 2))
  plt.imshow(255 - obs)
  plt.savefig(filename)


def anim(env, T=100, filename='/tmp/anim.mp4',
         show_clicks=True, show_resets=True):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  from matplotlib import animation

  fig = plt.figure(figsize=(2, 2))

  init_state = env.reset()
  states = [255 - init_state['image']]
  # positions = [init_state['disentangled'][:, :2]]

  resets = []
  actions = []
  for i in range(T):
    a = env.action_space.sample()
    state, _, done, _ = env.step(a)
    states.append(255 - state['image'])
    # positions.append(state['disentangled'][:, :2])
    resets.append(done)
    actions.append(a)

  im = plt.imshow(states[0], cmap=plt.get_cmap('jet'), vmin=0, vmax=255)

  def map_unit_point_to_image_range(point):
    """Assumes point in [0, 1]^2 and image has x- and y-lim of (-.5, 19.5)"""
    assert isinstance(point, Iterable) and len(point) == 2, 'bad point'
    point = (point[0], 1. - point[1])  # flip axes for comptability with imshow
    point = tuple(np.array(point) * 20. - .5)  # rescale to correct
    return point

  def updatefig(j):

    state, reset = states[j], resets[j]
    if show_resets and reset:  # this frame is terminal and env will reset next
      state = 255 - state  # indicate a reset by flashing inverse colors
    im.set_array(state)
    if show_clicks:  # hacking to overlay the click action as a scatter plot
      action = actions[j]
      print('>', end='', flush=True)
      ax = im.get_figure().get_axes()[0]
      xlim, ylim = ax.get_xlim(), ax.get_ylim()
      ax.clear()
      action = map_unit_point_to_image_range(action)
      ax.scatter(*action, s=50, c='k', marker='x')
      ax.set_xlim(xlim)
      ax.set_ylim(ylim)
      ax.add_image(im)

    return [im]

  print('animating')
  ani = animation.FuncAnimation(fig, updatefig, frames=T, interval=75,
                                repeat_delay=1000)
  # Set up formatting for the movie files
  Writer = animation.writers['ffmpeg']
  writer = Writer(fps=FPS, metadata=dict(artist='Me'), bitrate=1800)
  ani.save(filename, writer=writer)
  print('done')  # flush line
  # return ani.to_html5_video()
  return


def anim_with_attn(env, attn_mech, thresh, T=100,
                   filename='/tmp/anim_with_attn.mp4',
                   show_clicks=True, show_resets=True,
                   fps=FPS):
  """Show rollouts with attention under random policy."""
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  from matplotlib import animation

  num_sprites = len(env._env.state()['sprites'])
  gradient_colors = list(Color("red").range_to(Color("blue"),
                                               num_sprites))
  gradient_colors = [
    # tuple((np.array(gradient_color.get_rgb()) * 255).astype(np.int_))
    tuple(np.array(gradient_color.get_rgb()))
    for gradient_color in gradient_colors
  ]

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 2))
  xtick_labels = ['Sh{}'.format(i + 1) for i in range(num_sprites)]
  ytick_labels = xtick_labels + ['Act']
  ax2.set_xticks(np.arange(0, 4 * num_sprites, 4))
  ax2.set_xticklabels(xtick_labels, fontsize=8)
  ax2.set_yticks(np.arange(0, 5 * num_sprites, 4))
  ax2.set_yticklabels(ytick_labels, fontsize=8)

  for xlabel, ylabel, color in zip(
      ax2.get_xticklabels()[:num_sprites],
      ax2.get_yticklabels()[:num_sprites],
      gradient_colors
  ):
    xlabel.set_color(color)
    ylabel.set_color(color)

  plt.tight_layout()

  # set title
  for i, gradient_color in enumerate(gradient_colors):
    shape_id = i + 1
    fig.text(0.1 + i * .25, 0.9, "Shape %d" % shape_id,
             ha="center", va="bottom", size="small",
             weight='bold',
             color=gradient_color)


  init_state = env.reset()
  states = [255 - init_state['image']]

  def get_mask(state, action):
    state = torch.tensor(state.ravel())
    action = torch.tensor(action)
    state = state.unsqueeze(0)  # expand batch dim
    action = action.unsqueeze(0)  # expand batch dim
    model_input = torch.cat((state, action), -1)
    device = 'cpu'
    model_input = model_input.to(device)

    with torch.no_grad():
      _, mask, _ = attn_mech.forward_with_mask(model_input)
      # add dummy columns for (state, action -> next action) portion
      mask = mask.squeeze()
      # dummy_columns = torch.zeros(len(mask), 2)
      # mask = torch.cat((mask, dummy_columns), -1)
    mask = mask.cpu().numpy()
    mask = (mask > thresh).astype(np.float32)
    return mask

  resets = []
  actions = []
  masks = []
  for i in range(T):
    a = env.action_space.sample()
    state, _, done, _ = env.step(a)
    states.append(255 - state['image'])
    # positions.append(state['disentangled'][:, :2])
    resets.append(done)
    actions.append(a)
    mask = get_mask(state['disentangled'], a)
    masks.append(mask)

  im1 = ax1.imshow(states[0], cmap=plt.get_cmap('jet'), vmin=0, vmax=255)
  im2 = ax2.imshow(masks[0], cmap=plt.get_cmap('jet'), vmin=0, vmax=255)

  def map_unit_point_to_image_range(point):
    """Assumes point in [0, 1]^2 and image has x- and y-lim of (-.5, 19.5)"""
    assert isinstance(point, Iterable) and len(point) == 2, 'bad point'
    point = (point[0], 1. - point[1])  # flip axes for comptability with imshow
    point = tuple(np.array(point) * 20. - .5)  # rescale to correct
    return point

  def updatefig(j):

    state, reset = states[j], resets[j]
    if show_resets and reset:  # this frame is terminal and env will reset next
      state = 255 - state  # indicate a reset by flashing inverse colors
    im1.set_array(state)
    if show_clicks:  # hacking to overlay the click action as a scatter plot
      action = actions[j]
      print('>', end='', flush=True)
      ax = im1.get_figure().get_axes()[0]
      xlim, ylim = ax.get_xlim(), ax.get_ylim()
      ax.clear()
      action = map_unit_point_to_image_range(action)
      ax.scatter(*action, s=50, c='k', marker='x')
      ax.set_xlim(xlim)
      ax.set_ylim(ylim)
      ax.add_image(im1)

    mask = masks[j]
    mask = (255 * mask).astype(np.int_)
    im2.set_array(mask)
    return [im1, im2]

  print('animating')
  ani = animation.FuncAnimation(fig, updatefig, frames=T, interval=75,
                                repeat_delay=1000)
  # Set up formatting for the movie files
  Writer = animation.writers['ffmpeg']
  writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
  ani.save(filename, writer=writer)
  print('done')  # flush line
  return
