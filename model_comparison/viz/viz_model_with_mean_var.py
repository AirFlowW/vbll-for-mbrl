from typing import Tuple
import torch
import matplotlib.pyplot as plt

from vbll.layers.regression import VBLLReturn

"""
model output either (mean, variance) or VBLLReturn object
"""
def viz_model(model, dataloader, stdevs = 1., title = None, save_path=None, dataset: Tuple[torch.tensor,torch.tensor] = None):
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

  plt.plot(Xp, Y_mean)
  plt.fill_between(Xp, Y_mean - stdevs * Y_stdev, Y_mean + stdevs * Y_stdev, alpha=0.2, color='b')
  plt.fill_between(Xp, Y_mean - 2 * stdevs * Y_stdev, Y_mean + 2 * stdevs * Y_stdev, alpha=0.2, color='b')
  if dataset is not None:
    (X, Y) = dataset
  else:
    (X, Y) = dataloader.dataset.X, dataloader.dataset.Y
  plt.scatter(X, Y, color='k')
  plt.axis([-1.5, 1.5, -2, 2])
  if not title == None:
    plt.title(title)

  if save_path is not None:
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
  else:
    plt.show(block=True)

  return Y_mean, Y_stdev, Xp