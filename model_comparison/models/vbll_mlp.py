import vbll

import torch
import torch.nn as nn
import numpy as np

class VBLLMLP(nn.Module):
  """
  An MLP model with a VBLL last layer.

  cfg: a config containing model parameters.
  """

  def __init__(self, cfg):
    super(VBLLMLP, self).__init__()

    self.params = nn.ModuleDict({
        'in_layer': nn.Linear(cfg.IN_FEATURES, cfg.HIDDEN_FEATURES),
        'core': nn.ModuleList([nn.Linear(cfg.HIDDEN_FEATURES, cfg.HIDDEN_FEATURES) for i in range(cfg.NUM_LAYERS)]),
        'out_layer': vbll.Regression(cfg.HIDDEN_FEATURES, cfg.OUT_FEATURES, cfg.REG_WEIGHT, parameterization=cfg.PARAM, 
                                     cov_rank=cfg.COV_RANK, prior_scale = cfg.PRIOR_SCALE, wishart_scale = cfg.WISHART_SCALE)
        })
    self.in_activation = nn.ELU()
    self.activations = nn.ModuleList([nn.ELU() for i in range(cfg.NUM_LAYERS)])
    self.cfg = cfg

  def forward(self, x):
    x = self.in_activation(self.params['in_layer'](x))

    for layer, ac in zip(self.params['core'], self.activations):
      x = ac(layer(x))

    return self.params['out_layer'](x)

def train_vbll(dataloader, model, train_cfg, verbose = True):
  """Train a VBLL model."""

  # We explicitly list the model parameters and set last layer weight decay to 0
  # This isn't critical but can help performance.
  param_list = [
      {'params': model.params.in_layer.parameters(), 'weight_decay': train_cfg.WD},
      {'params': model.params.core.parameters(), 'weight_decay': train_cfg.WD},
      {'params': model.params.out_layer.parameters(), 'weight_decay': 0.}
  ]

  optimizer = train_cfg.OPT(param_list,
                            lr=train_cfg.LR,
                            weight_decay=train_cfg.WD)

  for epoch in range(train_cfg.NUM_EPOCHS + 1):
    model.train()
    running_loss = []

    for train_step, (x, y) in enumerate(dataloader):
      optimizer.zero_grad()
      out = model(x)
      loss = out.train_loss_fn(y) # note we use the output of the VBLL layer for the loss

      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.CLIP_VAL)
      optimizer.step()
      running_loss.append(loss.item())

    if epoch % train_cfg.VAL_FREQ == 0 and verbose:
      print('Epoch: {:4d},  loss: {:10.4f}'.format(epoch, np.mean(running_loss)))
      running_loss = []

class train_cfg_vbll:
  NUM_EPOCHS = 1000
  BATCH_SIZE = 32
  LR = 1e-3
  WD = 1e-4
  OPT = torch.optim.AdamW
  CLIP_VAL = 1
  VAL_FREQ = 100

class cfg_vbll:
    def __init__(self, dataset_length, parameterization = 'dense', cov_rank = None):
        self.REG_WEIGHT = 1./dataset_length
        self.PARAM = parameterization
        self.COV_RANK = cov_rank
    IN_FEATURES = 1
    HIDDEN_FEATURES = 64
    OUT_FEATURES = 1
    NUM_LAYERS = 4
    PRIOR_SCALE = 1.
    WISHART_SCALE = .1