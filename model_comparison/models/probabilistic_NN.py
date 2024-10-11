import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class ProbabilisticNN(nn.Module):
  """
  A standard MLP regression model.

  cfg: a config containing model parameters.
  """
  def __init__(self, cfg):
    super(ProbabilisticNN, self).__init__()

    # define model layers
    self.params = nn.ModuleDict({
        'in_layer': nn.Linear(cfg.IN_FEATURES, cfg.HIDDEN_FEATURES),
        'core': nn.ModuleList([nn.Linear(cfg.HIDDEN_FEATURES, cfg.HIDDEN_FEATURES) for i in range(cfg.NUM_LAYERS)]),
        'mean_layer': nn.Linear(cfg.HIDDEN_FEATURES, cfg.OUT_FEATURES),
        'covariance_layer': nn.Linear(cfg.HIDDEN_FEATURES, cfg.OUT_FEATURES)
        })

    # ELU activations are an arbitrary choice
    self.activation = nn.ELU()
    self.activations = nn.ModuleList([self.activation for i in range(cfg.NUM_LAYERS)])
    self.cfg = cfg

  def forward(self, x):
    x = self.activation(self.params['in_layer'](x))
    
    for layer, ac in zip(self.params['core'], self.activations):
      x = ac(layer(x))

    return self.params['mean_layer'](x), self.params['covariance_layer'](x)
  
def nll_loss(mean, var, y_true):
    """Compute the negative log likelihood loss (mean for multiple datapoints)."""
    positive_covariance = torch.clamp(var, min=1e-6)
    negative_log_likelihood = 0.5 * torch.log(positive_covariance) + \
        (0.5 * ((y_true - mean)**2)) / positive_covariance
    return torch.mean(negative_log_likelihood)

def train(dataloader, model, train_cfg, verbose = True):
  """Train a standard regression model with NLL loss."""
  loss_fn = nll_loss

  param_list = model.parameters()
  optimizer = train_cfg.OPT(param_list,
                            lr=train_cfg.LR,
                            weight_decay=train_cfg.WD)

  for epoch in range(train_cfg.NUM_EPOCHS + 1):
    model.train()
    running_loss = []

    for train_step, (x, y) in enumerate(dataloader):
      optimizer.zero_grad()

      mean, var = model(x)
      loss = loss_fn(mean, var, y)

      loss.backward()
      optimizer.step()

      running_loss.append(loss.item())
    if epoch % train_cfg.VAL_FREQ == 0 and verbose:
      print('Epoch {} loss: {:.3f}'.format(epoch, np.mean(running_loss)))
      running_loss = []

# -------- config --------
class train_cfg_pnn:
  NUM_EPOCHS = 1000
  BATCH_SIZE = 32
  LR = 3e-3
  WD = 0.
  OPT = torch.optim.AdamW
  VAL_FREQ = 100

class cfg_pnn:
    IN_FEATURES = 1
    HIDDEN_FEATURES = 64
    OUT_FEATURES = 1
    NUM_LAYERS = 4