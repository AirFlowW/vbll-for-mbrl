import torch
import torch.nn as nn
from model_comparison.models import probabilistic_NN_with_softplus_for_var as pnn

"""
unused as this is now done by the general default gaussian_ensemble
"""

class ProbabilisticEnsemble(nn.Module):

  def __init__(self, cfg_member, cfg_ensemble):
    super(ProbabilisticEnsemble, self).__init__()

    self.params = nn.ModuleDict({
        'members': nn.ModuleList([pnn.ProbabilisticNN(cfg_member) 
                                  for i in range(cfg_ensemble.NUM_MEMBERS)]),
        })
    self.cfg_ensemble = cfg_ensemble

  def forward(self, x):
    means = torch.stack([model(x)[0] for model in self.params['members']])
    variances = torch.stack([model(x)[1] for model in self.params['members']])

    
    ensemble_mean = torch.mean(means, dim=0)
    inner = variances + torch.pow(means, 2)
    ensemble_variance = torch.mean(inner, dim=0) - torch.pow(ensemble_mean, 2)
  

    return ensemble_mean, ensemble_variance
  
def train(dataloader, model, train_cfg, verbose = True):
  for i, probabilistic_model in enumerate(model.params['members']):
      if verbose:
        print(f"Train model member {i}")
      pnn.train(dataloader, probabilistic_model, train_cfg, verbose = False)

# -------- config --------
class train_cfg_member:
  NUM_EPOCHS = 1000
  BATCH_SIZE = 32
  LR = 1e-3
  WD = 0.
  OPT = torch.optim.AdamW
  VAL_FREQ = 100

class cfg_member:
    IN_FEATURES = 1
    HIDDEN_FEATURES = 64
    OUT_FEATURES = 1
    NUM_LAYERS = 4

class cfg_ensemble:
    NUM_MEMBERS = 4


"""We used the entire training dataset to train each network since deep NNs typically perform better with more data, 
although it is straightforward to use a random subsample if need be. We found that random initialization of the NN parameters, 
along with random shuffling of the data points, was sufficient to obtain good performance in practice."""