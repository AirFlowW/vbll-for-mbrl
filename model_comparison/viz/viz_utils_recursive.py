import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.lines import Line2D

def init_plot(path="model_comparison/viz/scientific_style.mplstyle"):
    plt.style.use(path)

def set_size(width_pt=397, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to sit nicely in our document.

    Parameters
    ----------
    width_pt: float, optional, default: 245 (IEEE template)
            Document width in points
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5 ** .5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def plot(datasets, names, max_figures_per_row=1, dataloader = None, save_path_wo_ending = None, legend_space = 0.4):
    """
    Creates subplots arranged in a grid for each dataset in 'datasets'. Each dataset should be structured as:
        - A list of groups, where each group is a list of runs (seeds), and each run is a list of loss values per epoch.
    
    Parameters:
    -----------
    datasets : list
        A list of datasets. Each dataset is a list of groups, and each group contains lists (runs with different seeds).
    names : list
        A list of names for the groups. These names (and their order) are assumed to be identical across all datasets.
    max_figures_per_row : int, optional
        Maximum number of subplots (datasets) per row.
    
    The function plots, for each dataset, one line per group (mean loss across seeds) with a shaded error region
    (mean ± standard deviation across seeds). A single global legend is placed outside the subplots at the top
    center, arranged in one row.
    """

    n_plots = len(datasets)
    ncols = max_figures_per_row
    nrows = math.ceil(n_plots / max_figures_per_row)

    init_plot()
    x_size, subplots_height = set_size(subplots=(nrows,ncols))
    y_size = subplots_height + legend_space
    
    # Create the subplots grid
    fig, axs = plt.subplots(nrows, ncols, figsize=(x_size, y_size), squeeze=False)
    axs = axs.flatten()  # Flatten for easy iteration.

    font_size = 11
    
    # Determine the number of groups (assumed identical across datasets)
    n_groups = len(datasets[0]) if n_plots > 0 else 0
    
    # Use the default color cycle from the current mplstyle.
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Plot each subplot.
    for i, dataset in enumerate(datasets):
        ax = axs[i]
        model_pred = False
        if i == 1 and dataloader is not None:
            (X, Y) = dataloader.dataset.X, dataloader.dataset.Y
            plt.scatter(X, Y, color='k', s=15)

        for j, group in enumerate(dataset):
            if len(group) == 3:
                mean = group[0]
                std = group[1]
                upper_deviation = mean + std
                lower_deviation = mean - std
                x = group[2]
                model_pred = True
                y_label = "Prediction"
                x_label = "Input"
            else:
                # Convert list of seeds to a NumPy array: shape (num_seeds, num_epochs)
                percentile = 5
                if percentile is not None:
                    group_array = np.array(group)
                    mean = np.mean(group_array, axis=0)
                    max = np.max(group_array)
                    print(f"max: {max}")
                    lower_deviation = np.percentile(group_array, percentile, axis=0)
                    upper_deviation = np.percentile(group_array, 100-percentile, axis=0)
                    
                else:
                    deviation = np.std(group_array, axis=0)
                    upper_deviation = mean + deviation
                    lower_deviation = mean - deviation

                x = np.arange(1, len(mean) + 1)
                y_label = "Loss"
                x_label = "Epoch"
            
            # Use the color from the current style; if there are more groups than colors, cycle through.
            color = default_colors[j % len(default_colors)]
            ax.plot(x, mean, color=color)
            ax.fill_between(x, lower_deviation, upper_deviation,
                            color=color, alpha=0.2, lw=0)
            if model_pred:
                ax.axis([-1.5, 1.5, -2, 2])
        
        # ax.set_title(f'Dataset {i+1}')
        ax.grid(False)
        # Remove axis labels as requested.
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
    
    # Remove any extra axes (if the grid is larger than the number of datasets)
    for k in range(n_plots, len(axs)):
        fig.delaxes(axs[k])
    
    
    # Adjust layout to avoid overlap.
    fig.tight_layout()
    # Make room at the top for the legend.
    fig.subplots_adjust(top=subplots_height/y_size)
    # Create custom legend handles using the default style colors.
    legend_handles = [
        Line2D([], [], color=default_colors[j % len(default_colors)], label=names[j])
        for j in range(n_groups)
    ]
    # Place the global legend outside the figure (above all subplots) in one row.
    fig.legend(handles=legend_handles,
               loc='upper center',
            #    bbox_to_anchor=(.9, 0.8),
               ncol=n_groups,
               frameon=False,
               fontsize=font_size)
    
    if save_path_wo_ending:
        plt.savefig(save_path_wo_ending + '.pdf', bbox_inches='tight')
        plt.close()
    else:
        plt.show()



# Example usage:
if __name__ == '__main__':
    # Dummy data for demonstration.
    # Each dataset has two groups; each group has three seeds and each seed has 10 epochs.
    
    # Dataset 1
    dataset1 = [
        # Group 1
        [
            [0.9, 0.8, 0.75, 0.70, 0.68, 0.66, 0.65, 0.64, 0.63, 0.62],
            [0.92, 0.81, 0.76, 0.71, 0.69, 0.67, 0.66, 0.65, 0.64, 0.63],
            [0.88, 0.79, 0.74, 0.69, 0.67, 0.65, 0.64, 0.63, 0.62, 0.61]
        ],
        # Group 2
        [
            [1.1, 1.0, 0.95, 0.90, 0.87, 0.85, 0.83, 0.82, 0.81, 0.80],
            [1.12, 1.02, 0.97, 0.92, 0.89, 0.87, 0.85, 0.84, 0.83, 0.82],
            [1.08, 0.98, 0.93, 0.88, 0.85, 0.83, 0.81, 0.80, 0.79, 0.78]
        ]
    ]
    
    # Dataset 2
    dataset2 = [
        # Group 1
        [
            [0.85, 0.80, 0.77, 0.73, 0.70, 0.69, 0.68, 0.67, 0.66, 0.65],
            [0.87, 0.82, 0.78, 0.74, 0.71, 0.70, 0.69, 0.68, 0.67, 0.66],
            [0.84, 0.79, 0.76, 0.72, 0.69, 0.68, 0.67, 0.66, 0.65, 0.64]
        ],
        # Group 2
        [
            [1.05, 1.00, 0.96, 0.93, 0.90, 0.88, 0.87, 0.86, 0.85, 0.84],
            [1.07, 1.02, 0.98, 0.95, 0.92, 0.90, 0.89, 0.88, 0.87, 0.86],
            [1.03, 0.98, 0.94, 0.91, 0.88, 0.86, 0.85, 0.84, 0.83, 0.82]
        ]
    ]
    
    # Combine the datasets into a list.
    datasets = [dataset1, dataset2]
    
    # Define the names for the groups (same for all datasets).
    group_names = ['Configuration A', 'Configuration B']
    
    # Call the plotting function.
    plot(datasets, group_names)

