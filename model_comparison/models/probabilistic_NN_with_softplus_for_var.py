import torch
import torch.nn as nn
from model_comparison.default_train import default_train

class ProbabilisticNN(nn.Module):

  def __init__(self, cfg):
    super(ProbabilisticNN, self).__init__()

    # define model layers
    self.params = nn.ModuleDict({
        'in_layer': nn.Linear(cfg.IN_FEATURES, cfg.HIDDEN_FEATURES),
        'core': nn.ModuleList([nn.Linear(cfg.HIDDEN_FEATURES, cfg.HIDDEN_FEATURES) for i in range(cfg.NUM_LAYERS)]),
        'mean_layer': nn.Linear(cfg.HIDDEN_FEATURES, cfg.OUT_FEATURES),
        'covariance_layer': nn.Linear(cfg.HIDDEN_FEATURES, cfg.OUT_FEATURES)
        })

    self.activation = nn.Tanh()
    self.activations = nn.ModuleList([self.activation for i in range(cfg.NUM_LAYERS)])
    self.cfg = cfg

  def forward(self, x):
    x = self.activation(self.params['in_layer'](x))
    
    for layer, ac in zip(self.params['core'], self.activations):
      x = ac(layer(x))

    mean = self.params['mean_layer'](x)
    covariance = torch.nn.functional.softplus(self.params['covariance_layer'](x))

    return mean, covariance
  
def nll_loss(mean_var: tuple, y_true):
    """Compute the negative log likelihood loss (mean for multiple datapoints)."""
    mean, var = mean_var
    negative_log_likelihood = 0.5 * (torch.log(var) + (((y_true - mean)**2) / var))
    return torch.mean(negative_log_likelihood)

def train(dataloader, model, train_cfg, verbose = True):
  default_train(dataloader,model,train_cfg,nll_loss, verbose)

# -------- config --------
class train_cfg_pnn:
  NUM_EPOCHS = 1000
  BATCH_SIZE = 32
  LR = 1e-3
  WD = 0.
  OPT = torch.optim.AdamW
  VAL_FREQ = 100

class cfg_pnn:
    IN_FEATURES = 1
    HIDDEN_FEATURES = 64
    OUT_FEATURES = 1
    NUM_LAYERS = 4