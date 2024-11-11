from model_comparison.models.pnn import ProbabilisticNN, train
from model_comparison.models.vbll_mlp import VBLLMLP, train_vbll


def pre_train(dataloader, model, train_cfg, verbose=False):
    if isinstance(model, VBLLMLP):
        train_vbll(dataloader, model, train_cfg, verbose=verbose)
    if isinstance(model, ProbabilisticNN):
        train(dataloader, model, train_cfg, verbose=verbose)