

import torch

from vbll.layers.regression import VBLLReturn


def viz_pred(model):
  """Visualize model predictions, including predictive uncertainty."""
  model.eval()
  X = torch.linspace(-1.5, 1.5, 1000)[..., None]
  Xp = X.detach().numpy().squeeze()

  pred = model(X)
  if isinstance(pred, tuple):
    Y_mean, Y_stdev = pred
  elif isinstance(pred, VBLLReturn):
    try:
        Y_mean = pred.predictive.mean
        Y_stdev = pred.predictive.covariance
    except:
      raise ValueError("model output must be either (mean, variance) or VBLLReturn object")
  elif isinstance(pred, torch.Tensor):
    Y_mean = pred
    Y_stdev = torch.zeros_like(pred)

  
  Y_mean = Y_mean.detach().numpy().squeeze()
  Y_stdev = torch.sqrt(Y_stdev.squeeze()).detach().numpy()

  return Y_mean, Y_stdev, Xp
