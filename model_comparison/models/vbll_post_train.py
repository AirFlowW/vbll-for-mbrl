import vbll

import time
import torch
import numpy as np
from model_comparison.models import mlp

def post_train_vbll(dataloader, model, post_train_cfg, verbose = True):
    """Post Training a VBLL model."""

    feature_extractor = mlp.MLP(mlp.cfg_mlp)
    mlp_train_cfg = mlp.train_cfg_mlp
    mlp_train_cfg.NUM_EPOCHS = post_train_cfg.NUM_EPOCHS_TO_TRAIN_FEATURES
    mlp.train(dataloader, feature_extractor, mlp_train_cfg, verbose = False)

    model.params['in_layer'] = feature_extractor.params['in_layer']
    model.params['core'] = feature_extractor.params['core']

    for param in model.params['in_layer'].parameters():
        param.requires_grad = False
    
    for param in model.params['core'].parameters():
        param.requires_grad = False

    param_list = [
        {'params': model.params.out_layer.parameters(), 'weight_decay': 0.}
    ]

    optimizer = post_train_cfg.OPT(param_list,
                            lr=post_train_cfg.LR,
                            weight_decay=post_train_cfg.WD)

    start = time.perf_counter()

    for epoch in range(post_train_cfg.NUM_EPOCHS_TO_TRAIN_LL + 1):
        model.train()
        running_loss = []

        for train_step, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            out = model(x)
            loss = out.train_loss_fn(y) # note we use the output of the VBLL layer for the loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), post_train_cfg.CLIP_VAL)
            optimizer.step()
            running_loss.append(loss.item())

        if epoch % post_train_cfg.VAL_FREQ == 0 and verbose:
            print('Epoch: {:4d},  loss: {:10.4f}'.format(epoch, np.mean(running_loss)))
            running_loss = []
            
    end = time.perf_counter()
    print(f"Post training took {end - start} seconds")

class post_train_cfg:
  NUM_EPOCHS_TO_TRAIN_LL = 1000
  NUM_EPOCHS_TO_TRAIN_FEATURES = 1000
  BATCH_SIZE = 32
  LR = 1e-3
  WD = 1e-4
  OPT = torch.optim.AdamW
  CLIP_VAL = 1
  VAL_FREQ = 100

class cfg_vbll:
    def __init__(self, dataset_length):
        self.REG_WEIGHT = 1./dataset_length
    IN_FEATURES = 1
    HIDDEN_FEATURES = 64
    OUT_FEATURES = 1
    NUM_LAYERS = 4
    PARAM = 'dense'
    PRIOR_SCALE = 1.
    WISHART_SCALE = .1