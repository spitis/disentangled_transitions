import numpy as np
from sklearn.linear_model import LinearRegression
from spriteworld import action_spaces


class SeededSelectBounce(action_spaces.SelectBounce):
  def __init__(self, seed=None, noise_scale=0.01, prevent_intersect=0.1):
    super(SeededSelectBounce, self).__init__(
      noise_scale=noise_scale, prevent_intersect=prevent_intersect)
    self.seed(seed)

  def seed(self, seed=None):
    np.random.seed(seed)


class LinearModelBasedSelectBounce(SeededSelectBounce):
  """Swaps spriteworld environment dynamics with regressor learned from data."""
  def __init__(self, dataset, seed=None, noise_scale=0.01,
               prevent_intersect=0.1):
    super(LinearModelBasedSelectBounce, self).__init__(
      seed=seed,
      noise_scale=noise_scale,
      prevent_intersect=prevent_intersect
    )
    # fit model
    data = dataset.data
    X = np.vstack(
      [np.hstack((state.ravel(), action.ravel()))
       for state, action, _, _ in data]
    )
    y = np.stack(
      [next_state.ravel() for _, _, _, next_state in data]
    )
    self.model = LinearRegression().fit(X, y).predict

  def step(self, action, sprites, *unused_args, **unused_kwargs):
    """Take an action and move the sprites.
    Args:
      action: Numpy array of shape (2,) in [0, 1].
      sprites: Iterable of sprite.Sprite() instances.
    Returns:
      Scalar cost of taking this action.
    """
    state = np.vstack(
      [np.hstack((sprite.x, sprite.y, sprite.velocity))
       for sprite in sprites]
    )  # N_sprites x 4 matrix
    model_input = np.hstack((state.ravel(), action.ravel()))  # flatten + concat
    model_input = model_input.reshape(1, -1)  # expand dimension for sklearn
    next_state = self.model(model_input)  # flattened predicted next state
    next_state = next_state.reshape(state.shape)  # unflatten
    for sprite, sprite_next_state in zip(sprites, next_state):
      position, velocity = sprite_next_state[:2], sprite_next_state[2:]
      sprite._position = position
      sprite._velocity = velocity

    return 0.


class NeuralModelBasedSelectBounce(SeededSelectBounce):
  def __init__(self, dataset, seed=None, noise_scale=0.01,
               prevent_intersect=0.1):
    super(NeuralModelBasedSelectBounce, self).__init__(
      seed=seed,
      noise_scale=noise_scale,
      prevent_intersect=prevent_intersect
    )
    # fit model
    # TODO(creager): split off validation set
    # TODO(creager): implement me
    raise NotImplementedError


# TODO(creager): spec out and implement an autoregressive baseline
