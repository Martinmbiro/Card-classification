import torch, numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score

# del torch, F, f1_score, accuracy_score

# function for model training
def train_batches(model: torch.nn.Module, train_dl: torch.utils.data.DataLoader,
                  optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module, device: str) -> tuple[float, float, float]:
  """
  Trains model on all batches of the training set DataLoader and returns
  average training loss, accuracy, and F1 score.

  Parameters
  ----------
  model : torch.nn.Module
      The model being trained.
  train_dl : torch.utils.data.DataLoader
      DataLoader for training data.
  optimizer : torch.optim.Optimizer
      The optimizer.
  loss_fn : torch.nn.Module
      Function used to calculate loss.
  device : str
      The device on which computation occurs.

  Returns
  -------
  tuple
      A tuple containing:
          - ls (float): Average training loss across all batches.
          - acc (float): Average training accuracy across all batches.
          - f1 (float): Average training F1 score across all batches.
  """
  # for reproducibility
  torch.manual_seed(0)
  torch.cuda.manual_seed(0)
  ls, acc, f1 = 0, 0, 0

  # training mode
  model.train()

  for x, y in train_dl:
      # move x, y to device
      x, y = x.to(device), y.to(device)
      # zero_grad
      optimizer.zero_grad()

      # forward pass
      logits = model(x)
      y_pred = F.softmax(logits, dim=1).argmax(dim=1).cpu().numpy()

      # loss
      loss = loss_fn(logits, y)
      # accumulate values
      ls += loss.item()
      acc += accuracy_score(y_true=y.cpu().numpy(), y_pred=y_pred)
      f1 += f1_score(y_true=y.cpu().numpy(),
                     y_pred=y_pred,
                     average='macro')

      # back propagation
      loss.backward()
      # optimizer step
      optimizer.step()

  # compute averages
  ls /= len(train_dl)
  acc /= len(train_dl)
  f1 /= len(train_dl)

  # return values
  return ls, acc, f1

def test_batches(model: torch.nn.Module, val_dl: torch.utils.data.DataLoader,
                 loss_fn: torch.nn.Module, device: str) -> tuple[float, float, float]:
  """
  Evaluates model on all batches of the test set DataLoader and returns
  average test loss, accuracy, and F1 score.

  Parameters
  ----------
  model : torch.nn.Module
      The model being evaluated.
  test_dl : torch.utils.data.DataLoader
      DataLoader for test data.
  loss_fn : torch.nn.Module
      Function used to calculate loss.
  device : str
      The device on which computation occurs.

  Returns
  -------
  tuple
      A tuple containing:
          - ls (float): Average test loss across all batches.
          - acc (float): Average test accuracy across all batches.
          - f1 (float): Average test F1 score across all batches.
  """
  ls, f1, acc = 0, 0, 0

  # evaluation-mode
  model.eval()

  with torch.inference_mode():
    for x, y in val_dl:
        # move x, y to device
        x, y = x.to(device), y.to(device)

        # forward pass
        logits = model(x)
        y_pred = F.softmax(logits, dim=1).argmax(dim=1).cpu().numpy()

        # loss
        loss = loss_fn(logits, y)

        # accumulate values
        ls += loss.item()
        acc += accuracy_score(y_true=y.cpu().numpy(), y_pred=y_pred)
        f1 += f1_score(y_true=y.cpu().numpy(),
                       y_pred=y_pred,
                       average='macro')

  # compute averages
  ls /= len(val_dl)
  acc /= len(val_dl)
  f1 /= len(val_dl)

  # return values
  return ls, acc, f1


def true_preds_proba(model: torch.nn.Module, test_dl: torch.utils.data.DataLoader,
                     device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """
  A function that returns true labels, predictions, and prediction probabilities
  from the passed DataLoader.

  Parameters
  ----------
  model : torch.nn.Module
      A neural network that subclasses torch.nn.Module.
  test_dl : torch.utils.data.DataLoader
      A DataLoader for the test dataset.
  device : str
      The device on which computation occurs.

  Returns
  -------
  tuple
      A tuple containing:
          - y_true (np.ndarray): A numpy array with true labels.
          - y_pred (np.ndarray): A numpy array with predicted labels.
          - y_proba (np.ndarray): A numpy array with predicted probabilities.
  """
  # empty lists
  y_true, y_preds, y_proba = list(), list(), list()
  with torch.inference_mode():
      model.eval()  # set eval mode
      for x, y in test_dl:
          # move x to device
          x = x.to(device)

          # make prediction
          logits = model(x)

          # prediction and probabilities
          proba = F.softmax(logits, dim=1)
          pred = F.softmax(logits, dim=1).argmax(dim=1)

          # append
          y_preds.append(pred)
          y_proba.append(proba)
          y_true.append(y)

  y_preds = torch.concatenate(y_preds).cpu().numpy()
  y_proba = torch.concatenate(y_proba).cpu().numpy()
  y_true = torch.concatenate(y_true).numpy()

  return y_true, y_preds, y_proba
