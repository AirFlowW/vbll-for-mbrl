from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import torch
import matplotlib.pyplot

from model_comparison.viz.viz_utils import init_plot, set_size
from vbll.layers.regression import VBLLReturn

def viz_ensemble(ensemble_model, dataloader, stdevs = 1., title = None, save_path=None, ax = None, default_colors = None, Thompson = True):
    if ax is not None:
        plt = ax
    else:
        plt = matplotlib.pyplot
        init_plot("model_comparison/viz/scientific_style_thompson.mplstyle")
        x_size, subplots_height = set_size(subplots=(1,1))
        y_size = subplots_height
        plt.figure(figsize=(x_size,y_size))
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        fontsize = 12
    

    X = torch.linspace(-1.5, 1.5, 1000)[..., None]
    Xp = X.detach().numpy().squeeze()

    if isinstance(ensemble_model, list):
        iterator = ensemble_model
        not_ensemble = True
    else:
        iterator = ensemble_model.params['members']
        not_ensemble = False

    for i, model in enumerate(iterator):
        model.eval()
        pred = model(X)
        if isinstance(pred, tuple):
            Y_mean, Y_stdev = pred
        elif isinstance(pred, VBLLReturn):
            try:
                Y_mean = pred.predictive.mean
                Y_stdev = pred.predictive.covariance
            except:
                raise ValueError("model output must be either (mean, variance) or VBLLReturn object")
        elif isinstance(pred, torch.Tensor):
            Y_mean = pred
            Y_stdev = torch.zeros_like(pred)
        
        Y_mean = Y_mean.detach().numpy().squeeze()
        Y_stdev = torch.sqrt(Y_stdev.squeeze()).detach().numpy()
        
        if Thompson:
            plt.plot(Xp, Y_mean)
        else:
            color = default_colors[i]
            plt.plot(Xp, Y_mean, color=color, alpha=0.9)
            plt.fill_between(Xp, Y_mean - stdevs * Y_stdev, Y_mean + stdevs * Y_stdev, alpha=0.3, color=color, lw=0)

        # for more visual uncertainty region
        if not_ensemble:
            plt.fill_between(Xp, Y_mean - 2 * stdevs * Y_stdev, Y_mean + 2 * stdevs * Y_stdev, alpha=0.2, color=color, lw=0)
    # Plot the actual data points from the dataloader
    zorder = 0 if not_ensemble else 20
    if Thompson:
        plt.scatter(dataloader.dataset.X, dataloader.dataset.Y, color='k', label='Data', zorder=zorder)
    else:
        plt.scatter(dataloader.dataset.X, dataloader.dataset.Y, color='k', label='Data', s=10, zorder=zorder)
    
    plt.axis([-1.5, 1.5, -2, 2])

    # legend_handles = [Patch(facecolor='none', edgecolor='none', label='Thompson Head:')]
    # names = ["...", "49", "50"]
    # legend_handles += [
    #     Line2D([], [], color=default_colors[j % len(default_colors)], label=names[j])
    #     for j in range(len(names))
    # ]
    # Place the global legend outside the figure (above all subplots) in one row.
    # plt.legend(handles=legend_handles,
    #            loc='upper center',
    #         #    bbox_to_anchor=(.9, 0.8),
    #            ncol=len(names)+1,
    #            frameon=False,
    #            fontsize=font_size)

    if Thompson:
        plt.xlabel("Input", fontsize=fontsize)
        plt.ylabel("Prediction", fontsize=fontsize)
    
    if not ax:
        plt.tight_layout()

    if title is not None:
        plt.title(title)
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    # else:
    #     plt.show(block=True)

    return