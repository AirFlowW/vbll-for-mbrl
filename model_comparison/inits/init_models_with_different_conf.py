
from model_comparison.utils.model_run_config_class import model_run_config
from model_comparison.models import pnn as probabilistic_NN, vbll_mlp
from model_comparison.test_config import cfg_sub_test
from model_comparison.viz import viz_model_with_mean_var as viz_w_var
from model_comparison.functions_to_evaluate import forward_test

BATCH_SIZE_OVERWRITE = 32
EPOCHS_TO_OVERWRITE = 5000
METHOD_TO_EVALUATE_OVERWRITE = forward_test.forward_pass

def init_vbllmlps(cfgs: list, datasets, different_cfgs: dict) -> list:
    """gives multiple vbllmlps to train on one dataset.
    Each vbllmlp has a different configuration.
    """
    models_to_train = []
    name_extensions = [item for sublist in different_cfgs.values() for item in sublist]
    
    train_cfg = vbll_mlp.train_cfg_vbll
    overwrite_train_cfg(train_cfg)
    
    for i, cfg in enumerate(cfgs):
        models_to_train.append(model_run_config(f"VBLLMLP--{name_extensions[i]}", train_cfg,
            [vbll_mlp.VBLLMLP(cfg(len(d))) for d in datasets], 
            METHOD_TO_EVALUATE_OVERWRITE, viz_w_var.viz_model, cfg_sub_test(False, None)))
        
    return models_to_train

def init_pnns(cfgs: list, datasets, different_cfgs: dict) -> list:
    
    models_to_train = []
    name_extensions = [item for sublist in different_cfgs.values() for item in sublist]

    train_cfg = probabilistic_NN.train_cfg_pnn
    overwrite_train_cfg(train_cfg)

    for i, cfg in enumerate(cfgs): 
        models_to_train.append(model_run_config(f"PNN--{name_extensions[i]}", train_cfg,
            [probabilistic_NN.ProbabilisticNN(cfg) for d in datasets], 
            METHOD_TO_EVALUATE_OVERWRITE, viz_w_var.viz_model, cfg_sub_test(False, None)))
        
    return models_to_train


def overwrite_train_cfg(train_cfg):
    setattr(train_cfg, "BATCH_SIZE", BATCH_SIZE_OVERWRITE)
    setattr(train_cfg, "NUM_EPOCHS", EPOCHS_TO_OVERWRITE)