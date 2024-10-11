import torch
import matplotlib.pyplot as plt

"""
model output either (mean, variance) or VBLLReturn object
"""
def viz_model(model, dataloader, stdevs = 1., title = None, save_path=None):
  """Visualize model predictions, including predictive uncertainty."""
  model.eval()
  X = torch.linspace(-1.5, 1.5, 1000)[..., None]
  Xp = X.detach().numpy().squeeze()

  pred = model(X)
  if isinstance(pred, tuple):
    Y_mean, Y_stdev = pred
  else:
    try:
        Y_mean = pred.predictive.mean
        Y_stdev = pred.predictive.covariance
    except:
      raise ValueError("model output must be either (mean, variance) or VBLLReturn object")
  
  Y_mean = Y_mean.detach().numpy().squeeze()
  Y_stdev = torch.sqrt(Y_stdev.squeeze()).detach().numpy()

  plt.plot(Xp, Y_mean)
  plt.fill_between(Xp, Y_mean - stdevs * Y_stdev, Y_mean + stdevs * Y_stdev, alpha=0.2, color='b')
  plt.fill_between(Xp, Y_mean - 2 * stdevs * Y_stdev, Y_mean + 2 * stdevs * Y_stdev, alpha=0.2, color='b')
  plt.scatter(dataloader.dataset.X, dataloader.dataset.Y, color='k')
  plt.axis([-1.5, 1.5, -2, 2])
  if not title == None:
    plt.title(title)

  if save_path is not None:
    plt.savefig(save_path)
    plt.close()
  else:
    plt.show()