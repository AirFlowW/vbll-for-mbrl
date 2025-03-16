import copy
import random
from torch import nn
import torch

from model_comparison.dataset import SimpleFnDataset
from model_comparison.models import mlp, vbll_mlp
from model_comparison.utils.seed import set_seed
from torch.utils.data import DataLoader
from model_comparison.models import default_gaussian_mean_var_ensemble as ensemble
from model_comparison.viz import viz_ensemble, viz_model_with_mean_var

from model_comparison.utils.general import path_to_save_plots
import os

if not os.path.exists(path_to_save_plots):
    os.makedirs(path_to_save_plots)

set_seed(42)
# different seeds
no_seeds = 1
seeds = []
for _ in range(no_seeds):
    seeds.append(random.randint(0, 10000000))
set_seed(seeds[0])

# different datasets
no_datasets = 1
no_samples = 256
datasets = []
for _ in range(no_datasets):
    datasets.append(SimpleFnDataset(num_samples=no_samples))

# model log diag zeros -> later with different seeds
model_noise_zeros = vbll_mlp.VBLLMLP(vbll_mlp.cfg_vbll(no_samples, 'dense_precision', None, 'zeros'))

# thompson models
no_of_thompson_heads = 50

train_cfg = vbll_mlp.train_cfg_vbll
for dataset in datasets:
    dataloader = DataLoader(dataset, batch_size=train_cfg.BATCH_SIZE, shuffle=True)
    vbll_mlp.train_vbll(dataloader, model_noise_zeros, vbll_mlp.train_cfg_vbll, verbose=False)

thompson_members = []
in_layer = copy.deepcopy(model_noise_zeros.params['in_layer'])
core = copy.deepcopy(model_noise_zeros.params['core'])

thompson_heads = model_noise_zeros.params['out_layer'].create_thompson_heads(no_of_thompson_heads)
for thompson_head in thompson_heads:
    weight_tensor = thompson_head
    linear_layer = nn.Linear(in_features=weight_tensor.shape[1], out_features=weight_tensor.shape[0])
    with torch.no_grad():
        linear_layer.weight = nn.Parameter(weight_tensor)
    linear_layer.bias.data.zero_()
    thompson_members.append(mlp.MLP(mlp.cfg_mlp, in_layer, core, linear_layer))

thompson_members = nn.ModuleList(thompson_members)
thompson_ensemble = ensemble.GaussianEnsemble(mlp.cfg_mlp, ensemble.cfg_ensemble, mlp.MLP, mlp.train, thompson_members)

# viz
viz_name = 'Thompson Ensemble'
viz_ensemble.viz_ensemble(thompson_ensemble, dataloader, save_path=path_to_save_plots + f'{viz_name}_members.pdf')
viz_model_with_mean_var.viz_model(thompson_ensemble, dataloader, title=viz_name, save_path=path_to_save_plots + f'{viz_name}.png')

viz_model_with_mean_var.viz_model(model_noise_zeros, dataloader, title='VBLL MLP', save_path=path_to_save_plots + 'vbll_mlp.png')