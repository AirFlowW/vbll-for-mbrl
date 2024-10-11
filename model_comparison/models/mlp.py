import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model_comparison.default_train import default_train

class MLP(nn.Module):
  """
  A standard MLP regression model.

  cfg: a config containing model parameters.
  """
  def __init__(self, cfg):
    super(MLP, self).__init__()

    # define model layers
    self.params = nn.ModuleDict({
        'in_layer': nn.Linear(cfg.IN_FEATURES, cfg.HIDDEN_FEATURES),
        'core': nn.ModuleList([nn.Linear(cfg.HIDDEN_FEATURES, cfg.HIDDEN_FEATURES) for i in range(cfg.NUM_LAYERS)]),
        'out_layer': nn.Linear(cfg.HIDDEN_FEATURES, cfg.OUT_FEATURES)
        })

    # ELU activations are an arbitrary choice
    self.activation = nn.ELU()
    self.activations = nn.ModuleList([self.activation for i in range(cfg.NUM_LAYERS)])
    self.cfg = cfg

  def forward(self, x):
    x = self.activation(self.params['in_layer'](x))
    
    for layer, ac in zip(self.params['core'], self.activations):
      x = ac(layer(x))

    return self.params['out_layer'](x)

def train(dataloader, model, train_cfg, verbose = True):
  default_train(dataloader, model, train_cfg, nn.MSELoss(), verbose)

# -------- config --------
class train_cfg_mlp:
  NUM_EPOCHS = 1000
  BATCH_SIZE = 32
  LR = 3e-3
  WD = 0.
  OPT = torch.optim.AdamW
  VAL_FREQ = 100

class cfg_mlp:
    IN_FEATURES = 1
    HIDDEN_FEATURES = 64
    OUT_FEATURES = 1
    NUM_LAYERS = 4