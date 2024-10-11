import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

def default_train(dataloader: DataLoader, model: nn.Module, train_cfg, loss_fn, verbose = True):
  """Train a standard regression model with MSE loss."""

  param_list = model.parameters()
  optimizer = train_cfg.OPT(param_list,
                            lr=train_cfg.LR,
                            weight_decay=train_cfg.WD)

  for epoch in range(train_cfg.NUM_EPOCHS + 1):
    model.train()
    running_loss = []

    for train_step, (x, y) in enumerate(dataloader):
      optimizer.zero_grad()

      out = model(x) 
      loss = loss_fn(out, y) 

      loss.backward()
      optimizer.step()
      
      running_loss.append(loss.item())

    if epoch % train_cfg.VAL_FREQ == 0 and verbose:
      print('Epoch {} loss: {:.3f}'.format(epoch, np.mean(running_loss)))
      running_loss = []