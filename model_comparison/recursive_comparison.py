import os
import pickle
import random

from model_comparison.dataset import SimpleFnDataset
from model_comparison.models import vbll_mlp, vbll_recursive
from model_comparison.utils.seed import set_seed
from torch.utils.data import DataLoader

set_seed(42)
# different seeds
no_seeds = 1
seeds = []
for _ in range(no_seeds):
    seeds.append(random.randint(0, 10000000))

# different datasets
no_datasets = 1
no_samples = 256
datasets = []
for _ in range(no_datasets):
    datasets.append(SimpleFnDataset(num_samples=no_samples))

train_cfg = vbll_recursive.recursive_train_cfg

for dataset in datasets:
    for seed in seeds:
        set_seed(seed)
        dataloader = DataLoader(dataset, batch_size=train_cfg.BATCH_SIZE, shuffle=True)
        
        set_seed(seed)
        train_cfg.RECURSIVE_NUM_EPOCHS = 1
        model = vbll_mlp.VBLLMLP(vbll_recursive.cfg_vbll(dataset_length=no_samples))
        dataloader_for_recursive_train, dataloader_for_full_train, recursive_models_pred = vbll_recursive.recursive_train_vbll(dataloader, model, train_cfg, verbose=True, recursive_train = True)
        set_seed(seed)
        train_cfg.RECURSIVE_NUM_EPOCHS = 100
        model = vbll_mlp.VBLLMLP(vbll_recursive.cfg_vbll(dataset_length=no_samples))
        dataloader_for_recursive_train_, dataloader_for_full_train_, gradient_models_pred = vbll_recursive.recursive_train_vbll(dataloader, model, train_cfg, verbose=False, recursive_train = False, dataloader_for_recursive_train=dataloader_for_recursive_train, dataloader_for_full_train = dataloader_for_full_train)

        # for x_1,x_2 in zip(dataloader_for_recursive_train,dataloader_for_recursive_train_):
        #     assert x_1 == x_2
        # for x_1,x_2 in zip(dataloader_for_full_train,dataloader_for_full_train_):
        #     assert x_1 == x_2
        
        wd = os.getcwd()
        folder_name = "recursive"
        with open(f"{wd}/{folder_name}/dataloader_for_recursive_train.pkl", "wb") as f:
            pickle.dump(dataloader_for_recursive_train, f)
        with open(f"{wd}/{folder_name}/dataloader_for_full_train.pkl", "wb") as f:
            pickle.dump(dataloader_for_full_train, f)
        with open(f"{wd}/{folder_name}/recursive_models_pred.pkl", "wb") as f:
            pickle.dump(recursive_models_pred, f)
        with open(f"{wd}/{folder_name}/gradient_models_pred.pkl", "wb") as f:
            pickle.dump(gradient_models_pred, f)

