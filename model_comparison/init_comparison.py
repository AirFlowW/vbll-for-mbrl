import random

import numpy as np
from model_comparison.dataset import SimpleFnDataset
from model_comparison.models import vbll_mlp
from model_comparison.utils.seed import set_seed
from torch.utils.data import DataLoader

from model_comparison.utils.general import path_to_save_plots

from model_comparison.viz import viz_model_with_mean_var, viz_pred
from model_comparison.viz.viz_utils import plot

set_seed(42)
# different seeds
no_seeds = 10
seeds = []
for _ in range(no_seeds):
    seeds.append(random.randint(0, 10000000))

# different datasets
no_datasets = 1
no_samples = 20
datasets = []
for _ in range(no_datasets):
    datasets.append(SimpleFnDataset(num_samples=no_samples))

train_cfg = vbll_mlp.train_cfg_vbll

zero_noise_losses = [] # will be list of list of losses List(List(List(float)))
random_noise_losses = [] # dim: Dataset * Seed x train_cfg.NUM_EPOCHS / train_cfg.VAL_FREQ x train_cfg.VAL_FREQ
train_count = 0
max_loss_model_zero_value = -1
max_loss_model_random_value = -1
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
        
        # starting_loss_zero = np.mean(zero_noise_losses[-1][0])
        # if starting_loss_zero > max_loss_model_zero_value:
        #     max_loss_model_zero = model_noise_zeros
        #     max_loss_model_zero_value = starting_loss_zero

        # starting_loss_random = np.mean(random_noise_losses[-1][0])
        # if starting_loss_random > max_loss_model_random_value:
        #     max_loss_model_random = model_noise_random
        #     max_loss_model_random_value = starting_loss_random

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

pred_zero_noise = viz_model_with_mean_var.viz_model(model_noise_zeros, dataloader, title='VBLL MLP with zero noise',save_path=path_to_save_plots + 'VBLL_MLP_zero_noise.png')
viz_model_with_mean_var.viz_model(model_noise_random, dataloader, title='VBLL MLP with random noise', save_path=path_to_save_plots + 'VBLL_MLP_random_noise.png')

plot_data = [random_noise_losses, zero_noise_losses]
predictions = [viz_pred.viz_pred(model_noise_random), viz_pred.viz_pred(model_noise_zeros)]
path_to_save = path_to_save_plots + "init-comparison"
names = ["Random Initialization", "Fixed Initialization"]
plot([plot_data,predictions], names, dataloader=dataloader, max_figures_per_row=2, save_path_wo_ending=path_to_save)