# Description: This script compares the runtime of different models on different dataset sizes.
from collections import defaultdict
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from copy import deepcopy
import time

from model_comparison.dataset import SimpleFnDataset, viz_data
from model_comparison.utils.pre_train import pre_train
from model_comparison.utils.model_run_config_class import model_run_config
from model_comparison.inits.init_models_with_different_conf import init_pnns, init_vbllmlps
from model_comparison.models import mlp, pnn as probabilistic_NN, vbll_mlp, vbll_sngp, probabilistic_ensemble as pe
from model_comparison.models import default_gaussian_mean_var_ensemble as ensemble, vbll_ensemble as VBLLE, vbll_post_train
from model_comparison.viz import viz_model_with_mean_var as viz_w_var
from model_comparison.viz.no_viz import no_viz
from model_comparison.viz import viz_only_mean_model as viz_wo_var
from model_comparison.viz import viz_ensemble as viz_ensemble

from model_comparison.test_config import cfg_test, cfg_sub_test

import matplotlib.image as mpimg
import os
import math

# ---- init
cfg_t = cfg_test()
runtimes = {}
models_to_train = []

datasets = []
for num_samples in cfg_t.different_dataset_sizes:
    datasets.append(SimpleFnDataset(num_samples=num_samples))

curr_dir = os.getcwd()
path_to_save_plots = os.path.join(curr_dir,'model_comparison','plots','')
if not os.path.exists(path_to_save_plots):
    os.makedirs(path_to_save_plots)
# ---- end init

if cfg_t.show_dataset:
    for dataset in datasets:
        viz_data(dataset)

# Init train MLP model config
if cfg_t.train_mlp:
    models_to_train.append(model_run_config('MLP', mlp.train_cfg_mlp,
        [mlp.MLP(mlp.cfg_mlp) for d in datasets], 
        mlp.train, viz_wo_var.viz_model, cfg_sub_test(cfg_t.show_mlp, None)))

# Init train Probabilistic model config
if cfg_t.train_pnn:
    models_to_train.append(model_run_config('PNN', probabilistic_NN.train_cfg_pnn,
        [probabilistic_NN.ProbabilisticNN(probabilistic_NN.cfg_pnn) for d in datasets], 
        probabilistic_NN.train, viz_w_var.viz_model, cfg_sub_test(cfg_t.show_pnn, None)))

# Init train Probabilistic Ensemble model config
if cfg_t.train_pe:
    models_to_train.append(model_run_config('PE', pe.train_cfg_member,
        [ensemble.GaussianEnsemble(pe.cfg_member,pe.cfg_ensemble,probabilistic_NN.ProbabilisticNN, probabilistic_NN.train) for d in datasets], 
        ensemble.train, viz_w_var.viz_model, cfg_sub_test(cfg_t.show_pe, cfg_t.show_pe_members)))
    
# Init train vbll with different kl penaltys model config
"""What if the uncertainty doesn't match our expectations or goals?
There are several ways to control uncertainty within VBLL models.
The simplest and most effective method to control the scale of uncertainty that 
we have found is modifying the KL regularization weight, REG_WEIGHT.
We can train a different VBLL model with a larger REG_WEIGHT:"""
if cfg_t.train_vbll_kl:
    cfg = vbll_mlp.cfg_vbll
    new_cfgs = []
    for dataset in datasets:
        for kl_factor in cfg_t.vbll_kl_weight:
            new_cfg = deepcopy(cfg(dataset_length=len(dataset)))
            new_cfg.REG_WEIGHT *= kl_factor
            new_cfgs.append(new_cfg)
    
    for kl_factor in cfg_t.vbll_kl_weight:
        kl_factor_string = "VBLL-KL-" + str(kl_factor) + 'x'
        
        models_to_train.append(model_run_config(kl_factor_string, vbll_mlp.train_cfg_vbll,
            [vbll_mlp.VBLLMLP(cfg) for cfg in new_cfgs], 
            vbll_mlp.train_vbll, viz_w_var.viz_model, cfg_sub_test(cfg_t.show_vbll_kl, None)))
        
## different parameterizations of last layer
# Init train VBLL 'diagonal' model config
if cfg_t.train_vbll_diagonal:
    models_to_train.append(model_run_config('VBLL_diagonal', vbll_mlp.train_cfg_vbll,
            [vbll_mlp.VBLLMLP(vbll_mlp.cfg_vbll(dataset_length=len(dataset), parameterization='diagonal')) for dataset in datasets], 
            vbll_mlp.train_vbll, viz_w_var.viz_model, cfg_sub_test(cfg_t.show_vbll_diagonal, None)))
    
# Init train VBLL 'lowrank' model config
if cfg_t.train_vbll_lowrank:
    models_to_train.append(model_run_config('VBLL_lowrank', vbll_mlp.train_cfg_vbll,
            [vbll_mlp.VBLLMLP(vbll_mlp.cfg_vbll(dataset_length=len(dataset), parameterization='lowrank', cov_rank=1)) for dataset in datasets], 
            vbll_mlp.train_vbll, viz_w_var.viz_model, cfg_sub_test(cfg_t.show_vbll_lowrank, None)))
    
# Init train VBLL 'dense_precision' model config
if cfg_t.train_vbll_dense_precision:
    models_to_train.append(model_run_config('VBLL_dense_precision', vbll_mlp.train_cfg_vbll,
            [vbll_mlp.VBLLMLP(vbll_mlp.cfg_vbll(dataset_length=len(dataset), parameterization='dense_precision')) for dataset in datasets], 
            vbll_mlp.train_vbll, viz_w_var.viz_model, cfg_sub_test(cfg_t.show_vbll_dense_precision, None)))
## different parameterizations of last layer end ---
        
# Init train VBLL Post Train model config
if cfg_t.train_post_train:
    models_to_train.append(model_run_config('VBLL_POST_TRAIN', vbll_post_train.post_train_cfg,
            [vbll_mlp.VBLLMLP(vbll_mlp.cfg_vbll(dataset_length=len(dataset))) for dataset in datasets], 
            vbll_post_train.post_train_vbll, viz_w_var.viz_model, cfg_sub_test(cfg_t.show_vbll_kl, None)))


# Init train VBLL Ensemble model config
if cfg_t.train_vbll_e:
    models_to_train.append(model_run_config('VBLL_E', vbll_mlp.train_cfg_vbll,
        [ensemble.GaussianEnsemble(vbll_mlp.cfg_vbll(dataset_length=len(d)), VBLLE.cfg_ensemble , vbll_mlp.VBLLMLP, vbll_mlp.train_vbll) for d in datasets], 
        ensemble.train, viz_w_var.viz_model, cfg_sub_test(cfg_t.show_vbll_e, cfg_t.show_vbll_e_members)))

# Init train vbll sngp model config
if cfg_t.train_vbll_sngp:
    models_to_train.append(model_run_config('VBLL_SNGP', vbll_sngp.train_cfg_vbll_sngp,
        [vbll_sngp.SNVResMLP(vbll_sngp.cfg_vbll_sngp(dataset_length=len(d))) for d in datasets], 
        vbll_mlp.train_vbll, viz_w_var.viz_model, cfg_sub_test(cfg_t.show_vbll_sngp, None)))
    
if cfg_t.evaluate_different_cfgs_vbll:
    main_cfg = vbll_mlp.cfg_vbll
    new_cfgs = []
    for key, item in cfg_t.different_cfgs.items():
        for value in item:
            new_cfg = deepcopy(main_cfg)
            setattr(new_cfg, key, value)
            new_cfgs.append(new_cfg)
    models_to_train += init_vbllmlps(new_cfgs, datasets, cfg_t.different_cfgs)

if cfg_t.evaluate_different_cfgs_pnn:
    main_cfg = probabilistic_NN.cfg_pnn
    new_cfgs = []
    for key, item in cfg_t.different_cfgs.items():
        for value in item:
            new_cfg = deepcopy(main_cfg)
            setattr(new_cfg, key, value)
            new_cfgs.append(new_cfg)
    models_to_train += init_pnns(new_cfgs, datasets, cfg_t.different_cfgs)
    

# main train, measure runtime and viz section
for model_run in models_to_train:
    model_name = model_run.model_name
    train_cfg = model_run.train_cfg
    models = model_run.models
    to_eval_fn = model_run.train_fn
    viz_model_fn = model_run.viz_model_fn
    cfg_sub_t = model_run.cfg_sub_t

    if not cfg_t.viz_of_models:
        viz_model_fn = no_viz

    print(f"Evaluate {model_name} model")
    runtimes[model_name] = []
    
    for k in range(len(models)-len(datasets)+1): # to support different configs of a model
        for i, dataset in enumerate(datasets):
            dataloader = DataLoader(dataset, batch_size=train_cfg.BATCH_SIZE, shuffle=True)

            model = models[i+k]
            if cfg_t.pre_train and not cfg_t.x_to_evaluate == 'dataset_size':
                pre_train(dataloader, model, train_cfg(), verbose = True)

            start = time.perf_counter()
            to_eval_fn(dataloader, model, train_cfg(), verbose = True)
            end = time.perf_counter()
            runtimes[model_name].append(end - start)

            if cfg_sub_t.show_members is not None and cfg_sub_t.show_members:
                viz_ensemble.viz_ensemble(model, dataloader, title=model_name)

            if cfg_sub_t.show_model:
                viz_model_fn(model, dataloader, title=model_name)
                
            if len(cfg_t.different_dataset_sizes) == i+1:
                viz_model_fn(model, dataloader, title=model_name, save_path=path_to_save_plots + model_name + '.png')
                if cfg_sub_t.show_members is not None:
                    viz_ensemble.viz_ensemble(model, dataloader,title=model_name, save_path=path_to_save_plots + model_name + '_members.png')

# Compare runtimes in graph form
if cfg_t.compare_times:
    saved_plots_wo_path = [key for key in runtimes.keys()]
    for saved_plot in saved_plots_wo_path:
        if saved_plot[-1] == 'E':
            saved_plots_wo_path.append(saved_plot + '_members')
    saved_plots = [path_to_save_plots + plot + '.png' for plot in saved_plots_wo_path]

    n = len(saved_plots)  

    plots_per_row = n
    num_rows = math.ceil(n / plots_per_row)
    fig, axs = plt.subplots(num_rows + 1, plots_per_row, figsize=(15, (num_rows + 1) * 5))

    for i, plot_file in enumerate(saved_plots):
        row = i // plots_per_row
        col = i % plots_per_row
        try:
            img = mpimg.imread(plot_file)  # Versuche, das Bild zu laden
            axs[row, col].imshow(img)      
            axs[row, col].axis('off')      
            axs[row, col].set_title(saved_plots_wo_path[i])
        except Exception as e:
            print(f"Error loading {plot_file}: {e}")
            axs[row, col].axis('off')

    # deactivate empty fields
    for j in range(i+1, plots_per_row * num_rows):
        row = j // plots_per_row
        col = j % plots_per_row
        axs[row, col].axis('off')

    # merge different cfgs to let them have one line in the resulting plot
    if cfg_t.x_to_evaluate == 'cfgs':
        merged_data = defaultdict(list)

        for key, values in runtimes.items():
            base_key = key.rsplit('--', 1)[0]
            merged_data[base_key].extend(values)

        runtimes = dict(merged_data)

    for key, values in runtimes.items():
        if values:
            if cfg_t.x_to_evaluate == 'cfgs':
                x_axis_key = ""
                for x_label_from_cfgs in cfg_t.different_cfgs.keys():
                    x_axis_key += x_label_from_cfgs
                    break
                axs[num_rows, 0].plot(cfg_t.different_cfgs[x_axis_key], values, label=key, marker='o')
            elif cfg_t.x_to_evaluate == 'dataset_size':
                axs[num_rows, 0].plot(cfg_t.different_dataset_sizes, values, label=key, marker='o')
            else:
                raise ValueError(f'Unknown x_to_evaluate: {cfg_t.x_to_evaluate}')

    for ax in axs[num_rows, 1:]:
        ax.remove()
        
    axs[num_rows, 0].set_position([0.125, 0.05, 0.775, 0.4])
    x_label = ""
    for key in cfg_t.different_cfgs.keys():
        x_label += key
    axs[num_rows, 0].set_xlabel(x_label)
    axs[num_rows, 0].set_ylabel('Runtime (seconds)')
    axs[num_rows, 0].set_title('Runtime Comparison Across Different Models and Dataset Sizes')
    axs[num_rows, 0].legend()
    axs[num_rows, 0].grid(True)

    plt.tight_layout()
    plt.savefig(path_to_save_plots + 'overall.png')
    plt.show()