import copy
from functools import partial
import logging

import numpy as np
from sklearn.linear_model import LinearRegression
from spriteworld import action_spaces
import torch

Tensor = torch.Tensor


class SeededSelectBounce(action_spaces.SelectBounce):
  def __init__(self, seed=None, noise_scale=0.01, prevent_intersect=0.1):
    super(SeededSelectBounce, self).__init__(
      noise_scale=noise_scale, prevent_intersect=prevent_intersect)
    self.seed(seed)

  def seed(self, seed=None):
    np.random.seed(seed)
    torch.manual_seed(seed)




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
  EVAL_EVERY = 5  # TODO(): make this a command line arugment?

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

    # fit model
    best_valid_loss = np.inf
    best_model = copy.deepcopy(model)
    consequtive_epochs_without_improvement = 0
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
            consequtive_epochs_without_improvement = 0
          else:
            consequtive_epochs_without_improvement += self.EVAL_EVERY
          if consequtive_epochs_without_improvement > patience_epochs:
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


# TODO(creager): spec out and implement an autoregressive baseline
