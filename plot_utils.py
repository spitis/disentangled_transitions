from typing import Iterable

import numpy as np


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
