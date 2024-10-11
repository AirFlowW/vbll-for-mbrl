import torch
import torch.nn as nn

class GaussianEnsemble(nn.Module):

  def __init__(self, cfg_member, cfg_ensemble, model_class, train_member_fn):
    super(GaussianEnsemble, self).__init__()

    self.params = nn.ModuleDict({
        'members': nn.ModuleList([model_class(cfg_member) 
                                  for i in range(cfg_ensemble.NUM_MEMBERS)]),
        })
    self.cfg_ensemble = cfg_ensemble
    self.train_member_fn = train_member_fn

  def forward(self, x):
    predictions = [model(x) for model in self.params['members']]
    if predictions and isinstance(predictions[0], tuple):
      means = torch.stack([pred[0] for pred in predictions])
      variances = torch.stack(([pred[1] for pred in predictions]))
    else:
      means = torch.stack([pred.predictive.mean.squeeze() for pred in predictions])
      variances = torch.stack([pred.predictive.covariance.squeeze() for pred in predictions])

    
    ensemble_mean = torch.mean(means, dim=0)
    inner = variances + torch.pow(means, 2)
    ensemble_variance = torch.mean(inner, dim=0) - torch.pow(ensemble_mean, 2)
  

    return ensemble_mean, ensemble_variance
  
def train(dataloader, model, train_cfg, verbose = True):
    for i, probabilistic_model in enumerate(model.params['members']):
       if verbose:
          print(f"Train model member {i}")
       model.train_member_fn(dataloader, probabilistic_model, train_cfg, verbose = False)

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