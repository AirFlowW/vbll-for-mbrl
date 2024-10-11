import torch
import matplotlib.pyplot as plt

def viz_ensemble(ensemble_model, dataloader, stdevs = 1., title = None, save_path=None):
    plt.figure(figsize=(10, 6))
    
    X = torch.linspace(-1.5, 1.5, 1000)[..., None]
    Xp = X.detach().numpy().squeeze()

    for i, model in enumerate(ensemble_model.params['members']):
        model.eval()
        pred = model(X)
        if isinstance(pred, tuple):
            Y_mean, Y_stdev = pred
        else:
            try:
                Y_mean = pred.predictive.mean
                Y_stdev = pred.predictive.covariance
            except:
                raise ValueError("model output must be either (mean, variance) or VBLLReturn object")
        
        Y_mean = Y_mean.detach().numpy().squeeze()
        Y_stdev = torch.sqrt(Y_stdev.squeeze()).detach().numpy()

        plt.plot(Xp, Y_mean, label=f'Model {i + 1}')
        plt.fill_between(Xp, Y_mean - stdevs * Y_stdev, Y_mean + stdevs * Y_stdev, alpha=0.2)

        # for more visual uncertainty region
        # plt.fill_between(Xp, Y_mean - 2 * stdevs * Y_stdev, Y_mean + 2 * stdevs * Y_stdev, alpha=0.2, color='b')
    
    # Plot the actual data points from the dataloader
    plt.scatter(dataloader.dataset.X, dataloader.dataset.Y, color='k', label='Data', zorder=5)
    
    plt.axis([-1.5, 1.5, -2, 2])
    
    if title is not None:
        plt.title(title)
    
    plt.legend()  # Add a legend to differentiate between models
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

    return
    for model in ensemble_model.params['members']:
        model.eval()
        X = torch.linspace(-1.5, 1.5, 1000)[..., None]
        Xp = X.detach().numpy().squeeze()

        pred = model(X)
        if isinstance(pred, tuple):
            Y_mean, Y_stdev = pred
        else:
            try:
                Y_mean = pred.predictive.mean
                Y_stdev = pred.predictive.covariance
            except:
                raise ValueError("model output must be either (mean, variance) or VBLLReturn object")
        
        Y_mean = Y_mean.detach().numpy().squeeze()
        Y_stdev = torch.sqrt(Y_stdev.squeeze()).detach().numpy()

        plt.plot(Xp, Y_mean)
        plt.fill_between(Xp, Y_mean - stdevs * Y_stdev, Y_mean + stdevs * Y_stdev, alpha=0.2, color='b')
        plt.fill_between(Xp, Y_mean - 2 * stdevs * Y_stdev, Y_mean + 2 * stdevs * Y_stdev, alpha=0.2, color='b')
        plt.scatter(dataloader.dataset.X, dataloader.dataset.Y, color='k')
        plt.axis([-1.5, 1.5, -2, 2])
        if not title == None:
            plt.title(title)

        if save_path is not None:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()