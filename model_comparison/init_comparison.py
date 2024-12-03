import random

import numpy as np
from model_comparison.dataset import SimpleFnDataset
from model_comparison.models import vbll_mlp
from model_comparison.utils.seed import set_seed
from torch.utils.data import DataLoader

from model_comparison.viz import viz_model_with_mean_var

set_seed(42)
# different seeds
no_seeds = 5
seeds = []
for _ in range(no_seeds):
    seeds.append(random.randint(0, 10000000))

# different datasets
no_datasets = 5
no_samples = 256
datasets = []
for _ in range(no_datasets):
    datasets.append(SimpleFnDataset(num_samples=no_samples))

# model log diag zeros -> later with different seeds
# model log diag random -> later with different seeds

train_cfg = vbll_mlp.train_cfg_vbll

zero_noise_losses = [] # will be list of list of losses List(List(List(float)))
random_noise_losses = [] # dim: Dataset * Seed x train_cfg.NUM_EPOCHS / train_cfg.VAL_FREQ x train_cfg.VAL_FREQ
train_count = 0
for dataset in datasets:
    for seed in seeds:
        set_seed(seed)
        dataloader = DataLoader(dataset, batch_size=train_cfg.BATCH_SIZE, shuffle=True)
        # zero noise init cfg
        set_seed(seed)
        model_noise_zeros = vbll_mlp.VBLLMLP(vbll_mlp.cfg_vbll(no_samples, 'dense', None, 'zeros'))
        zero_noise_losses.append(vbll_mlp.train_vbll(dataloader, model_noise_zeros, vbll_mlp.train_cfg_vbll, verbose=False))
        # random noise init cfg
        set_seed(seed)
        model_noise_random = vbll_mlp.VBLLMLP(vbll_mlp.cfg_vbll(no_samples, 'dense', None, 'random'))
        random_noise_losses.append(vbll_mlp.train_vbll(dataloader, model_noise_random, vbll_mlp.train_cfg_vbll, verbose=False))
        
        train_count += 1
        print(f'Finished training {train_count}/{no_datasets*no_seeds} models')

for i, dataset in enumerate(datasets):
    for j, seed in enumerate(seeds):
        print(f'Dataset {i} with {dataset.num_samples} samples and seed no. {j}, seed: {seed}')
        print('Starting losses:')
        print(f'Zero noise loss: {np.mean(zero_noise_losses[i+j][0])}')
        print(f'Random noise loss: {np.mean(random_noise_losses[i+j][0])}')
        print("")
print("")
print(f'Zero noise loss overall mean: {np.mean(zero_noise_losses)}')
print(f'Random noise loss overall mean: {np.mean(random_noise_losses)}')

viz_model_with_mean_var.viz_model(model_noise_zeros, dataloader, title='VBLL MLP with zero noise')