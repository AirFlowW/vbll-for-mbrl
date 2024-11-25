import time
from typing import Tuple
import torch
from torch.utils.data import DataLoader, TensorDataset
from model_comparison.utils.general import path_to_save_plots

from model_comparison.models import vbll_mlp
from model_comparison.viz.viz_model_with_mean_var import viz_model

def mask_to_seperate_datasets(data, threshold = 0):
    mask = data < threshold
    return mask

def mask_to_seperate_datasets_along_X_data(X, threshold = 0):
    return mask_to_seperate_datasets(X, threshold)

def mask_seperate_X_zth_greatest_values(X, z):
    return mask_to_seperate_datasets_along_X_data(X, zth_greatest_value(X, z))

def zth_greatest_value(tensor, z):
    if tensor.dim() != 2 or tensor.size(1) != 1:
        raise ValueError("Tensor shape must be (n,1).")
    
    if z < 1 or z > tensor.size(0):
        raise ValueError("z must be within 1 and n.")
    
    sorted_tensor, _ = torch.sort(tensor, descending=False, dim=0)
    return sorted_tensor[-z].item()

def get_val_loss_of_dataloader(dataloader, model):
    loss_sum = 0
    is_training = model.training
    model.eval()
    for train_step, (x, y) in enumerate(dataloader):
        out = model(x)
        loss = out.val_loss_fn(y)
        loss_sum += loss.item()
    if is_training:
        model.train()
    return loss_sum

# main function
def recursive_train_vbll(dataloader, model, recursive_train_cfg, verbose = True, point_for_recursive_train: Tuple[float,float] = None):
    """Training a VBLL model on particular data and then use recursive updates to train on more data.
    Possible use cases:
    - train on a subset of data and then train recursively on the rest of the data: use masks to choose which data
    can be X or Y related mask
    - train on a (certain: greatest zth data points regarding X coordinate) subset of data and then train recursively
    on the rest of the data: use masks greatest zth values
    - train on all data points and then add a single point via recursive training: use point_for_recursive_train
    (uncomment point_for_recursive_train = (X, Y))
    """

    # Single point adding for recursive training:
    # if a point is given only this point is used for recursive training
    # point_for_recursive_train = (0., -2.0)

    # split data into full train and recursive train sets
    X_all_data = []
    Y_all_data = []
    for x_batch, y_batch in dataloader:
        X_all_data.append(x_batch)
        Y_all_data.append(y_batch)
    X_all_data = torch.cat(X_all_data, dim=0)
    Y_all_data = torch.cat(Y_all_data, dim=0) # dataset_size x 1 tensor
    
    if point_for_recursive_train is None:
        threshold = 0
    else:
        threshold = 50000
    # mask = mask_to_seperate_datasets(Y_all_data, threshold)
    mask = mask_seperate_X_zth_greatest_values(X_all_data, 3)
    opposite_mask = ~mask
    
    X_for_full_train = X_all_data[mask.squeeze()]
    Y_for_full_train = Y_all_data[mask.squeeze()]
    if point_for_recursive_train is None:
        X_for_recursive_train = X_all_data[opposite_mask.squeeze()]
        Y_for_recursive_train = Y_all_data[opposite_mask.squeeze()]
    else:
        X_for_recursive_train = torch.tensor([[point_for_recursive_train[0]]])
        Y_for_recursive_train = torch.tensor([[point_for_recursive_train[1]]])

    
    dataset_for_full_train = TensorDataset(X_for_full_train, Y_for_full_train)
    dataset_for_recursive_train = TensorDataset(X_for_recursive_train, Y_for_recursive_train)
    
    fulltrain_dataset_size = len(torch.squeeze(X_for_full_train))
    dataloader_for_full_train = DataLoader(dataset_for_full_train,
                                        batch_size=recursive_train_cfg.BATCH_SIZE, shuffle=True)
    
    recursive_dataset_size = len(torch.squeeze(X_for_recursive_train))
    dataloader_for_recursive_train = DataLoader(dataset_for_recursive_train,
                                        batch_size=recursive_dataset_size, shuffle=True)
    
    model.params['out_layer'].regularization_weight = 1/fulltrain_dataset_size

    # start counting time
    start = time.perf_counter()
    # train full vbll model
    vbll_mlp.train_vbll(dataloader_for_full_train, model, recursive_train_cfg, verbose = verbose)

    # viz what full train did
    start_viz_timer = time.perf_counter()
    viz_model(model, dataloader_for_full_train, title="Recursive-Fulltrain", dataset=(X_for_full_train, Y_for_full_train),
              save_path=path_to_save_plots + "Recursive-Fulltrain" + '.png')
    viz_time = time.perf_counter() - start_viz_timer

    # train recursive vbll model
    recursive_train = True
    if not recursive_train:
        recursive_train_cfg.NUM_EPOCHS = recursive_train_cfg.RECURSIVE_NUM_EPOCHS
        vbll_mlp.train_vbll(dataloader_for_recursive_train, model, recursive_train_cfg, verbose = verbose)
    else:
        model.train()
        with torch.no_grad():
            if recursive_train_cfg.RECURSIVE_NUM_EPOCHS is not None:
                for epoch in range(recursive_train_cfg.RECURSIVE_NUM_EPOCHS):
                    for (x,y) in dataloader_for_recursive_train:
                        out = model(x)
                        out.train_loss_fn(y, recursive_update=True)
            else:
                recursive_iterations = 0
                last_loss_sum = get_val_loss_of_dataloader(dataloader, model)

                current_loss_sum = last_loss_sum
                while last_loss_sum >= current_loss_sum:
                    recursive_iterations += 1
                    last_loss_sum = current_loss_sum
                    for (x,y) in dataloader_for_recursive_train:
                        out = model(x)
                        out.train_loss_fn(y, recursive_update=True)
                    current_loss_sum = get_val_loss_of_dataloader(dataloader, model)
                if verbose:
                    print(f"Recursive iterations: {recursive_iterations}")


    end = time.perf_counter()
    return end - start - viz_time

class recursive_train_cfg:
  NUM_EPOCHS = int(1000)
  RECURSIVE_NUM_EPOCHS = 8
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
    PARAM = 'dense_precision'
    COV_RANK = None
    PRIOR_SCALE = 1.
    WISHART_SCALE = .1