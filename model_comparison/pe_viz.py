
import os
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from model_comparison.dataset import SimpleFnDataset
from model_comparison.models import pnn, probabilistic_ensemble as pe
from model_comparison.models import default_gaussian_mean_var_ensemble as ensemble
from model_comparison.utils.seed import set_seed
from model_comparison.viz import viz_model_with_mean_var as viz_w_var

from model_comparison.utils.general import path_to_save_plots
from model_comparison.viz import viz_ensemble
from model_comparison.viz.viz_utils import init_plot, set_size


if __name__ == "__main__":
    # init 
    set_seed(256)

    num_samples = 256
    dataset = SimpleFnDataset(num_samples=num_samples)

    if not os.path.exists(path_to_save_plots):
        os.makedirs(path_to_save_plots)


    # get pe model
    model = ensemble.GaussianEnsemble(pe.cfg_member,pe.cfg_ensemble,pnn.ProbabilisticNN, pnn.train)
    train_fn = ensemble.train
    train_cfg = pe.train_cfg_member

    dataloader = DataLoader(dataset, batch_size=train_cfg.BATCH_SIZE, shuffle=True)

    # train pe model
    train_fn(dataloader, model, train_cfg(), verbose = True)

    # init viz
    init_plot("model_comparison/viz/scientific_style_thompson.mplstyle")
    x_size, subplots_height = set_size(subplots=(1,2))
    y_size = subplots_height
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    font_size = 12

    fig, axs = plt.subplots(1, 2, figsize=(x_size, y_size), squeeze=False, sharey=True)
    axs = axs.flatten()  # Flatten for easy iteration.

    default_colors[0] = "#1F77B5"
    ax = axs[0]
    viz_ensemble.viz_ensemble([model], dataloader, ax=ax,default_colors=default_colors)
    ax.set_xlabel("Input",fontsize=font_size)
    ax.set_ylabel("Prediction",fontsize=font_size)

    ax=axs[1]
    viz_ensemble.viz_ensemble(model, dataloader, ax=ax,default_colors=default_colors[1:])
    ax.set_xlabel("Input",fontsize=font_size)
    ax.set_ylabel("")


    save_path = path_to_save_plots + 'pe_overview.pdf'
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.125)
    plt.savefig(save_path, bbox_inches='tight')