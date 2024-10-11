# Description: This script compares the runtime of different models on different dataset sizes.
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from copy import deepcopy
import time

from model_comparison.dataset import SimpleFnDataset, viz_data
from model_comparison.models import mlp, vbll_mlp, vbll_sngp, probabilistic_ensemble as pe, probabilistic_NN_with_softplus_for_var as probabilistic_NN
from model_comparison.models import default_gaussian_mean_var_ensemble as ensemble
from model_comparison.viz import viz_model_with_mean_var as viz_w_var
from model_comparison.viz import viz_only_mean_model as viz_wo_var
from model_comparison.viz import viz_ensemble as viz_ensemble

from model_comparison.test_config import cfg_test

import matplotlib.image as mpimg
import os
import math

cfg_t = cfg_test()
runtimes = {}

datasets = []
for num_samples in cfg_t.different_dataset_sizes:
    datasets.append(SimpleFnDataset(num_samples=num_samples))

curr_dir = os.getcwd()
path_to_save_plots = os.path.join(curr_dir,'model_comparison','plots','')
if not os.path.exists(path_to_save_plots):
    os.makedirs(path_to_save_plots)


if cfg_t.show_dataset:
    for dataset in datasets:
        viz_data(dataset)

# MLP model
if cfg_t.train_mlp:
    train_cfg = mlp.train_cfg_mlp
    runtimes['MLP'] = []
    print("Training MLP model")

    for i, dataset in enumerate(datasets):
        cfg = mlp.cfg_mlp
        dataloader = DataLoader(dataset, batch_size=train_cfg.BATCH_SIZE, shuffle=True)
        model = mlp.MLP(cfg())

        start = time.perf_counter()
        mlp.train(dataloader, model, train_cfg())
        end = time.perf_counter()
        runtimes['MLP'].append(end - start)
        
        if cfg_t.show_mlp:
            viz_wo_var.viz_model(model, dataloader,title='MLP.png')
        if len(cfg_t.different_dataset_sizes) == i+1:
            viz_wo_var.viz_model(model, dataloader, title='MLP.png', save_path=path_to_save_plots + 'MLP.png')

# Training Probabilistic model
if cfg_t.train_pnn:
    print("Training PNN model")
    runtimes['PNN'] = []
    train_cfg = probabilistic_NN.train_cfg_pnn
    cfg = probabilistic_NN.cfg_pnn

    for i, dataset in enumerate(datasets):
        dataloader = DataLoader(dataset, batch_size=train_cfg.BATCH_SIZE, shuffle=False)

        model = probabilistic_NN.ProbabilisticNN(cfg)

        start = time.perf_counter()
        probabilistic_NN.train(dataloader, model, train_cfg(), verbose = True)
        end = time.perf_counter()
        runtimes['PNN'].append(end - start)

        if cfg_t.show_vbll_kl:
            viz_w_var.viz_model(model, dataloader, title='PNN')
        if len(cfg_t.different_dataset_sizes) == i+1:
            viz_w_var.viz_model(model, dataloader,title='PNN', save_path=path_to_save_plots + 'PNN.png')

# VBLL model (and with different KL penalties)
"""What if the uncertainty doesn't match our expectations or goals?
There are several ways to control uncertainty within VBLL models.
The simplest and most effective method to control the scale of uncertainty that 
we have found is modifying the KL regularization weight, REG_WEIGHT.
We can train a different VBLL model with a larger REG_WEIGHT:"""
if cfg_t.train_vbll_kl:
    # for the individual KL penalties
    for kl_factor in cfg_t.vbll_kl_weight:
        kl_factor_string = str(kl_factor) + 'x'
        kl_factor_dict_string = "VBLL-KL-" + kl_factor_string
        print("Training VBLL model with KL penalties factor of: " + kl_factor_string)
        runtimes[kl_factor_dict_string] = []

        train_cfg = vbll_mlp.train_cfg_vbll
        cfg = vbll_mlp.cfg_vbll
        for i, dataset in enumerate(datasets):
            dataloader = DataLoader(dataset, batch_size=train_cfg.BATCH_SIZE, shuffle=True)
            new_cfg = deepcopy(cfg(dataset_length=len(dataset)))
            new_cfg.REG_WEIGHT *= kl_factor

            vbll_model = vbll_mlp.VBLLMLP(new_cfg)

            start = time.perf_counter()
            vbll_mlp.train_vbll(dataloader, vbll_model, train_cfg(), verbose = True)
            end = time.perf_counter()
            runtimes[kl_factor_dict_string].append(end - start)

            if cfg_t.show_vbll_kl:
                viz_w_var.viz_model(vbll_model, dataloader, title=f'{kl_factor_string} KL penalty')
            if len(cfg_t.different_dataset_sizes) == i+1:
                viz_w_var.viz_model(vbll_model, dataloader,
                    title=f'VBLL {kl_factor_string} KL penalty', 
                    save_path=path_to_save_plots + f'{kl_factor_dict_string}.png')

# VBLL model with SNGP
if cfg_t.train_vbll_sngp:
    print("Training VBLL SNGP model")
    runtimes['VBLL SNGP'] = []
    train_cfg = vbll_sngp.train_cfg_vbll_sngp

    for i, dataset in enumerate(datasets):
        cfg = vbll_sngp.cfg_vbll_sngp
        dataloader = DataLoader(dataset, batch_size=train_cfg.BATCH_SIZE, shuffle=True)

        snv_model = vbll_sngp.SNVResMLP(cfg(dataset_length=len(dataset)))

        start = time.perf_counter()
        vbll_mlp.train_vbll(dataloader, snv_model, train_cfg())
        end = time.perf_counter()
        runtimes['VBLL SNGP'].append(end - start)

        if cfg_t.show_vbll_sngp:
            viz_w_var.viz_model(snv_model, dataloader, title='SNGP.png')
        if len(cfg_t.different_dataset_sizes) == i+1:
            viz_w_var.viz_model(snv_model, dataloader, title='SNGP.png', save_path=path_to_save_plots + 'SNGP.png')


# Training Probabilistic Ensemble model
if cfg_t.train_pe:
    model_name = 'PE'
    print(f"Training {model_name} model")
    runtimes[model_name] = []
    train_cfg = pe.train_cfg_member
    cfg_member = pe.cfg_member
    cfg_ensemble = pe.cfg_ensemble

    for i, dataset in enumerate(datasets):
        dataloader = DataLoader(dataset, batch_size=train_cfg.BATCH_SIZE, shuffle=True)

        model = ensemble.GaussianEnsemble(cfg_member,cfg_ensemble,probabilistic_NN.ProbabilisticNN, probabilistic_NN.train)

        start = time.perf_counter()
        ensemble.train(dataloader, model, train_cfg(), verbose = True)
        end = time.perf_counter()
        runtimes[model_name].append(end - start)

        if cfg_t.show_pe_members:
            viz_ensemble.viz_ensemble(model, dataloader, title=model_name)

        if cfg_t.show_pe:
            viz_w_var.viz_model(model, dataloader, title=model_name)
            
        if len(cfg_t.different_dataset_sizes) == i+1:
            viz_w_var.viz_model(model, dataloader, title=model_name, save_path=path_to_save_plots + model_name + '.png')
            viz_ensemble.viz_ensemble(model, dataloader,title=model_name, save_path=path_to_save_plots + model_name + '_members.png')

# Compare runtimes in graph form
if cfg_t.compare_times:
    saved_plots_wo_path = [key for key in runtimes.keys()]
    if 'PE' in saved_plots_wo_path:
        saved_plots_wo_path.append('PE_members')
    saved_plots = [path_to_save_plots + plot + '.png' for plot in saved_plots_wo_path]

    n = len(saved_plots)  

    plots_per_row = n
    num_rows = math.ceil(n / plots_per_row)
    fig, axs = plt.subplots(num_rows + 1, plots_per_row, figsize=(15, (num_rows + 1) * 5))

    for i, plot_file in enumerate(saved_plots):
        row = i // plots_per_row
        col = i % plots_per_row
        img = mpimg.imread(plot_file)  
        axs[row, col].imshow(img)      
        axs[row, col].axis('off')      
        axs[row, col].set_title(saved_plots_wo_path[i])

    # deactivate empty fields
    for j in range(i+1, plots_per_row * num_rows):
        row = j // plots_per_row
        col = j % plots_per_row
        axs[row, col].axis('off')

    for key, values in runtimes.items():
        if values:
            axs[num_rows, 0].plot(cfg_t.different_dataset_sizes, values, label=key, marker='o')  # Plot in die letzte Zeile

    for ax in axs[num_rows, 1:]:
        ax.remove()  # Entfernen der leeren Subplots in der letzten Zeile
        
    axs[num_rows, 0].set_position([0.125, 0.05, 0.775, 0.4])  # Stretch Runtime-Plot über die gesamte Breite
    axs[num_rows, 0].set_xlabel('Dataset Size')
    axs[num_rows, 0].set_ylabel('Runtime (seconds)')
    axs[num_rows, 0].set_title('Runtime Comparison Across Different Models and Dataset Sizes')
    axs[num_rows, 0].legend()
    axs[num_rows, 0].grid(True)

    plt.tight_layout()
    plt.savefig(path_to_save_plots + 'overall.png')
    plt.show()