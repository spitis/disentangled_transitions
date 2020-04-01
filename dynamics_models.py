import copy
from functools import partial
import logging

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from spriteworld import action_spaces
import torch
import cmath

Tensor = torch.Tensor
Array = np.ndarray


def compute_test_loss(action_space, test_loader):
  """Compute test loss for a model-based action space."""

  if not hasattr(action_space, 'CONSUMES_TENSORS'):
    raise ValueError(
      'Received action space of bad type: %s' % type(action_space)
    )

  def _predict_next_state(state, action):
    """Predicts next state and returns as np array on the cpu."""
    if action_space.CONSUMES_TENSORS:
      with torch.no_grad():
        next_state = action_space.model(state, action)
        return next_state.cpu().numpy()
    else:
      state = state.cpu().numpy()
      action = action.cpu().numpy()
      return action_space.model(state, action)

  # accumulate predictions for all test examples
  next_states = []
  pred_next_states = []
  for state, action, next_state in test_loader:
    pred_next_state = _predict_next_state(state, action)
    pred_next_states.append(pred_next_state)
    next_state = next_state.cpu().numpy()  # for compatability with sklearn
    next_states.append(next_state)

  # compute errors from all predictions
  next_states = np.vstack(next_states)
  num_test_examples = len(next_states)
  next_states = next_states.reshape(num_test_examples, -1)  # flatten
  pred_next_states = np.vstack(pred_next_states)
  pred_next_states = pred_next_states.reshape(num_test_examples, -1)  # flatten
  test_loss = mean_squared_error(next_states, pred_next_states)
  return test_loss


class SeededSelectBounce(action_spaces.SelectBounce):
  def __init__(self, seed=None, noise_scale=0.01, prevent_intersect=0.1):
    super(SeededSelectBounce, self).__init__(
      noise_scale=noise_scale, prevent_intersect=prevent_intersect)
    self.seed(seed)

  def seed(self, seed=None):
    np.random.seed(seed)
    torch.manual_seed(seed)

class SeededSelectRedirect(SeededSelectBounce):

  def step(self, action, sprites, *unused_args, **unused_kwargs):
    """
    Causes sprite velocity to change direction.
    """
    if self._prevent_intersect > 0:
      barriers = sprites
    else:
      barriers = []
    sprite_poses = np.array([s.position for s in sprites])
    dists = np.linalg.norm(action[None] - sprite_poses, axis=1)
    closest_idx = np.argmin(dists)
    clicked_sprite = sprites[closest_idx]

    v_scale, v_angle = cmath.polar(complex(*clicked_sprite.velocity))
    z, target_v_angle = cmath.polar(complex(*(clicked_sprite.position - action)))
    new_v_angle = (1-z)*v_angle + z*target_v_angle

    new_v = cmath.rect(v_scale, new_v_angle)
    new_v = np.array([new_v.real, new_v.imag]) + np.random.normal(loc=[0., 0.], scale=self._noise_scale)
    new_v = new_v.astype(np.float32)  # for consistency with simulator and torch
    new_v = np.clip(new_v, -clicked_sprite._max_abs_vel, clicked_sprite._max_abs_vel)
    clicked_sprite._velocity = new_v

    for sprite in sprites:
      sprite.update_position(keep_in_frame=True, barriers=barriers, prevent_intersect=self._prevent_intersect, acted_on=(sprite is clicked_sprite))
    return 0.



class LinearModelBasedSelectBounce(SeededSelectBounce):
  """Swaps spriteworld environment dynamics with regressor learned from data."""
  CONSUMES_TENSORS = False

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
    regressor = LinearRegression().fit(X, y)
    self.model = partial(self.predict_next_state, regressor)

  @staticmethod
  def predict_next_state(model: LinearRegression,
                         state: Array,
                         action: Array):
    state_shape = state.shape
    batch_size = state_shape[0]
    state = state.reshape((batch_size, -1))  # flatten
    model_input = np.hstack((state, action))
    next_state = model.predict(model_input)  # flattened predicted next state
    next_state = next_state.reshape(state_shape)  # unflatten
    return next_state

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
    state = np.expand_dims(state, 0)  # expand batch dim
    action = np.expand_dims(action, 0)  # expand batch dim
    next_state = self.model(state, action)
    next_state = next_state.squeeze()  # remove batch dim
    for sprite, sprite_next_state in zip(sprites, next_state):
      position, velocity = sprite_next_state[:2], sprite_next_state[2:]
      sprite._position = position
      sprite._velocity = velocity

    return 0.


class NeuralModelBasedSelectBounce(SeededSelectBounce):
  """Swaps spriteworld environment dynamics with MLP learned from data."""
  EVAL_EVERY = 5  # TODO(): make this a command line arugment?
  CONSUMES_TENSORS = True

  def __init__(self,
               train_loader: torch.utils.data.DataLoader,
               valid_loader: torch.utils.data.DataLoader,
               num_hidden_units: int,
               activ_fn: torch.nn.Module,
               lr: float,
               num_epochs: int,
               patience_epochs: int,
               weight_decay: float,
               seed=None,
               noise_scale=0.01,
               prevent_intersect=0.1):
    super(NeuralModelBasedSelectBounce, self).__init__(
      seed=seed,
      noise_scale=noise_scale,
      prevent_intersect=prevent_intersect
    )

    # build model
    state_dim = np.prod(train_loader.dataset.s1.shape[1:])
    action_dim = train_loader.dataset.a.shape[-1]
    model = torch.nn.Sequential(
      torch.nn.Linear(state_dim + action_dim, num_hidden_units),
      activ_fn,
      torch.nn.Linear(num_hidden_units, state_dim)
    )

    # build optimizer
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    pred_criterion = torch.nn.MSELoss()
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(self.device)

    # fit model
    best_valid_loss = np.inf
    best_model = copy.deepcopy(model)
    consecutive_epochs_without_improvement = 0
    logging.info('Begin training!')
    for epoch in range(num_epochs):
      model.train()
      total_train_loss = 0.
      for state, action, next_state in train_loader:
        state = state.to(self.device)
        action = action.to(self.device)
        model.zero_grad()
        pred_next_state = self.predict_next_state(True, model, state, action)
        loss = pred_criterion(next_state.to(self.device), pred_next_state)
        loss.backward()
        opt.step()
        total_train_loss += loss.detach().item()
      if epoch % self.EVAL_EVERY == 0:
        # compute validation loss
        model.eval()
        total_valid_loss = 0.
        with torch.no_grad():
          for state, action, next_state in valid_loader:
            state = state.to(self.device)
            action = action.to(self.device)
            pred_next_state = self.predict_next_state(False, model, state,
                                                     action)
            total_valid_loss = total_valid_loss + pred_criterion(
              next_state.to(self.device), pred_next_state
            )
            total_train_loss += loss.item()

          logging.info('Ep {}. Tr Loss: {:.7f}. Va Loss: {:.7f}'.format(
            epoch,
            total_train_loss / len(train_loader),
            total_valid_loss / len(valid_loader)
          ))
          if total_valid_loss < best_valid_loss:
            best_valid_loss = total_valid_loss
            best_model = copy.deepcopy(model)
            consecutive_epochs_without_improvement = 0
          else:
            consecutive_epochs_without_improvement += self.EVAL_EVERY
          if consecutive_epochs_without_improvement > patience_epochs:
            logging.info('Stopping early after %d epochs' % epoch)
            break
    logging.info('End training!')

    # declare predict function by fixing model to best weights from training
    self.model = partial(self.predict_next_state, False, best_model)

  @staticmethod
  def predict_next_state(train: bool,
                         model: torch.nn.Module,
                         state: Tensor,
                         action: Tensor):
    batch_size = state.shape[0]
    state_shape = state.shape[1:]
    state = state.reshape(batch_size, -1)  # flatten
    model_inputs = torch.cat((state, action), -1)
    if train:  # trace gradients for trainig
      next_state = model(model_inputs)
    else:
      with torch.no_grad():  # eval mode
        next_state = model(model_inputs)
    next_state = next_state.reshape(-1, *state_shape)  # unflatten
    return next_state

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
    state_shape = state.shape
    state = state.reshape(1, -1)  # expand batch dim and flatten
    action = action.reshape(1, -1)  # expand batch dim
    state = torch.tensor(state).to(self.device)
    action = torch.tensor(action).to(self.device)
    next_state = self.model(state, action)  # flattened predicted next state
    next_state = next_state.numpy()
    next_state = next_state.reshape(state_shape)  # unflatten
    for sprite, sprite_next_state in zip(sprites, next_state):
      position, velocity = sprite_next_state[:2], sprite_next_state[2:]
      sprite._position = position
      sprite._velocity = velocity

    return 0.


class LSTMModelBasedSelectBounce(SeededSelectBounce):
  """Swaps spriteworld environment dynamics with LSTM learned from data."""
  EVAL_EVERY = 5  # TODO(): make this a command line arugment?
  NUM_LAYERS = 1  # TODO(): make this a command line arugment?
  BIAS = True  # TODO(): make this a command line arugment?
  BATCH_FIRST = True  # TODO(): make this a command line arugment?
  DROPOUT = 0.  # TODO(): make this a command line arugment?
  BIDIRECTIONAL = False  # cannot be true since we need causal predictions
  CONSUMES_TENSORS = True

  def __init__(self,
               train_loader: torch.utils.data.DataLoader,
               valid_loader: torch.utils.data.DataLoader,
               lr: float,
               num_epochs: int,
               patience_epochs: int,
               weight_decay: float,
               seed=None,
               noise_scale=0.01,
               prevent_intersect=0.1):
    super(LSTMModelBasedSelectBounce, self).__init__(
      seed=seed,
      noise_scale=noise_scale,
      prevent_intersect=prevent_intersect
    )

    # build model
    num_state_features = train_loader.dataset.s1.shape[-1]
    # NOTE: actions will be repeated as input to the LSTM at each step of seq
    num_action_features = train_loader.dataset.a.shape[-1]
    model = torch.nn.LSTM(
      input_size=num_state_features+num_action_features,
      hidden_size=num_state_features,
      num_layers=self.NUM_LAYERS,
      bias=self.BIAS,
      batch_first=self.BATCH_FIRST,
      dropout=self.DROPOUT,
      bidirectional=self.BIDIRECTIONAL,
    )

    # build optimizer
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    pred_criterion = torch.nn.MSELoss()
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(self.device)

   # fit model
    best_valid_loss = np.inf
    best_model = copy.deepcopy(model)
    consecutive_epochs_without_improvement = 0
    logging.info('Begin training!')
    # TODO(creager): refactor so that LSTMModelBasedSelectBounce and
    #  NeuralModelBasedSelectBounce share a train step to reduce lines of
    #  code and potential for bugs
    for epoch in range(num_epochs):
      model.train()
      total_train_loss = 0.
      for state, action, next_state in train_loader:
        state = state.to(self.device)
        action = action.to(self.device)
        model.zero_grad()
        pred_next_state = self.predict_next_state(True, model, state, action)
        loss = pred_criterion(next_state.to(self.device), pred_next_state)
        loss.backward()
        opt.step()
        total_train_loss += loss.detach().item()
      if epoch % self.EVAL_EVERY == 0:
        # compute validation loss
        model.eval()
        total_valid_loss = 0.
        with torch.no_grad():
          for state, action, next_state in valid_loader:
            state = state.to(self.device)
            action = action.to(self.device)
            pred_next_state = self.predict_next_state(False, model, state,
                                                     action)
            total_valid_loss = total_valid_loss + pred_criterion(
              next_state.to(self.device), pred_next_state
            )
            total_train_loss += loss.item()

          logging.info('Ep {}. Tr Loss: {:.7f}. Va Loss: {:.7f}'.format(
            epoch,
            total_train_loss / len(train_loader),
            total_valid_loss / len(valid_loader)
          ))
          if total_valid_loss < best_valid_loss:
            best_valid_loss = total_valid_loss
            best_model = copy.deepcopy(model)
            consecutive_epochs_without_improvement = 0
          else:
            consecutive_epochs_without_improvement += self.EVAL_EVERY
          if consecutive_epochs_without_improvement > patience_epochs:
            logging.info('Stopping early after %d epochs' % epoch)
            break
    logging.info('End training!')

    # declare predict function by fixing model to best weights from training
    self.model = partial(self.predict_next_state, False, best_model)

  @staticmethod
  def predict_next_state(train: bool,
                         model: torch.nn.Module,
                         state: Tensor,
                         action: Tensor):
    batch_size, num_sprites, num_state_features = state.shape
    # repeat actions as input to the LSTM at each step of seq
    action = action.unsqueeze(1).repeat(1, num_sprites, 1)
    model_inputs = torch.cat((state, action), -1)
    if train:  # trace gradients for trainig
      next_state, _ = model(model_inputs)
    else:
      with torch.no_grad():  # eval mode
        next_state, _ = model(model_inputs)
    return next_state

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
    state_shape = state.shape
    state = torch.tensor(state).to(self.device)
    action = torch.tensor(action).to(self.device)
    state = state.unsqueeze(0)  # expand batch dim
    action = action.unsqueeze(0)  # expand batch dim
    next_state = self.model(state, action)  # flattened predicted next state
    next_state = next_state.numpy()
    next_state = next_state.reshape(state_shape)  # unflatten
    for sprite, sprite_next_state in zip(sprites, next_state):
      position, velocity = sprite_next_state[:2], sprite_next_state[2:]
      sprite._position = position
      sprite._velocity = velocity

    return 0.
