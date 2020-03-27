def viz(obs, filename='./viz.pdf'):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt

  plt.figure(figsize=(2, 2))
  plt.imshow(255 - obs)
  plt.savefig(filename)


def anim(env, T=100, filename='/tmp/anim.mp4'):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  from matplotlib import animation

  fig = plt.figure(figsize=(2, 2))

  states = [255 - env.reset()['image']]

  for i in range(T):
    a = env.action_space.sample()
    state, _, _, _ = env.step(a)
    states.append(255 - state['image'])

  im = plt.imshow(states[0], cmap=plt.get_cmap('jet'), vmin=0, vmax=255)

  def updatefig(j):
    im.set_array(states[j])
    return [im]

  ani = animation.FuncAnimation(fig, updatefig, frames=T, interval=75,
                                repeat_delay=1000)
  # Set up formatting for the movie files
  Writer = animation.writers['ffmpeg']
  writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
  ani.save(filename, writer=writer)
  return ani.to_html5_video()
