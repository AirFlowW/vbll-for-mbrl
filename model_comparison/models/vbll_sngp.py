import vbll

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm
import numpy as np

class SNVResMLP(nn.Module):
  def __init__(self, cfg):
    super(SNVResMLP, self).__init__()
    self.cfg = cfg
    self.params = nn.ModuleDict({
        'in_layer': nn.Linear(cfg.IN_FEATURES, cfg.HIDDEN_FEATURES),
        'core': nn.ModuleList([spectral_norm(nn.Linear(cfg.HIDDEN_FEATURES, cfg.HIDDEN_FEATURES)) for i in range(cfg.NUM_LAYERS)]),
        'out_layer': vbll.Regression(cfg.RFF_FEATURES, cfg.OUT_FEATURES, cfg.REG_WEIGHT, prior_scale = cfg.PRIOR_SCALE, wishart_scale = cfg.WISHART_SCALE)
        })

    self.activations = nn.ModuleList([nn.ELU() for i in range(cfg.NUM_LAYERS-1)])
    self.W = torch.normal(torch.zeros(cfg.RFF_FEATURES, cfg.HIDDEN_FEATURES), cfg.KERNEL_SCALE * torch.ones(cfg.RFF_FEATURES, cfg.HIDDEN_FEATURES))
    self.b = torch.rand(cfg.RFF_FEATURES)*2*torch.pi

  def forward(self, x):
    x =  self.params['in_layer'](x)
    for layer, ac in zip(self.params['core'], self.activations):
      x = ac(layer(x)) + x

    x = torch.cos((self.W @ x[..., None]).squeeze(-1) + self.b)
    x = x * np.sqrt(2./self.cfg.RFF_FEATURES) * (self.cfg.KERNEL_SCALE)

    return self.params['out_layer'](x)

class train_cfg_vbll_sngp:
  NUM_EPOCHS = 1000
  BATCH_SIZE = 32
  LR = 3e-3
  WD = 0.
  OPT = torch.optim.AdamW
  CLIP_VAL = 1
  VAL_FREQ = 100

class cfg_vbll_sngp:
    def __init__(self, dataset_length):
        self.REG_WEIGHT = 1./dataset_length
    IN_FEATURES = 1
    HIDDEN_FEATURES = 64
    RFF_FEATURES = 128
    OUT_FEATURES = 1
    DROPOUT_RATE = 0.0
    NUM_LAYERS = 4
    PRIOR_SCALE = 1.
    WISHART_SCALE = .1
    KERNEL_SCALE = .5