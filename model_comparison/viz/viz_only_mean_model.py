import torch
import matplotlib.pyplot as plt

def viz_model(model, dataloader, title=None, save_path=None):
  """Visualize prediction of standard regression model."""
  model.eval()
  X = torch.linspace(-1.5, 1.5, 1000)[..., None]
  Y_pred = model(X)

  plt.plot(X.detach().numpy(), Y_pred.detach().numpy())
  plt.scatter(dataloader.dataset.X, dataloader.dataset.Y, color='k')
  plt.axis([-1.5, 1.5, -2, 2])
  if not title == None:
    plt.title(title)

  if save_path is not None:
    plt.savefig(save_path)
    plt.close()
  else:
    plt.show()