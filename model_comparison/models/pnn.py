import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class ProbabilisticNN(nn.Module):
    def __init__(self, cfg):
        super(ProbabilisticNN, self).__init__()
        
        layers = [nn.Linear(cfg.IN_FEATURES, cfg.HIDDEN_FEATURES), nn.ReLU()]
        
        for _ in range(cfg.NUM_LAYERS - 2):
            layers.append(nn.Linear(cfg.HIDDEN_FEATURES, cfg.HIDDEN_FEATURES))
            layers.append(nn.ReLU())
        
        self.mean_layer = nn.Linear(cfg.HIDDEN_FEATURES, cfg.OUT_FEATURES)
        self.var_layer = nn.Linear(cfg.HIDDEN_FEATURES, cfg.OUT_FEATURES)
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        features = self.network(x)
        mean = self.mean_layer(features)
        var = self.var_layer(features)
        var = torch.nn.functional.softplus(var)
        return mean, var

def nll_loss(mean, var, target):
    loss = 0.5 * torch.mean(torch.log(var) + ((target - mean) ** 2) / var)
    return loss

def train(dataloader: DataLoader, model: nn.Module, train_cfg, loss_fn = nll_loss, verbose=True):
    optimizer = train_cfg.OPT(model.parameters(), lr=train_cfg.LR, weight_decay=train_cfg.WD)
    
    for epoch in range(train_cfg.NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for batch_no, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()

            mean, log_var = model(inputs)
            loss = loss_fn(mean, log_var, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if verbose and (epoch + 1) % train_cfg.VAL_FREQ == 0:
            avg_loss = running_loss / len(dataloader)
            print(f'Epoch {epoch+1}, Loss: {avg_loss:.6f}')

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