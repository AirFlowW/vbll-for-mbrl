# Configure the test suite settings below.
# Specify which models to display and train, as well as the dataset size(s) to use.
# It is possible to define multiple dataset sizes for testing to see the runtime increase with increasing dataset size.
class cfg_test:
    show_dataset = False

    show_mlp = False

    show_vbll_kl = False
    show_vbll_diagonal = False
    show_vbll_lowrank = False
    show_vbll_dense_precision = False

    show_vbll_e = False
    show_vbll_e_members = False

    show_vbll_sngp = False
    show_recursive = False
    show_post_train = False

    show_pnn = False
    show_pe = False
    show_pe_members = False

    train_mlp = False
    
    train_vbll_kl = True
    vbll_kl_weight = [1] # 1 for normal ('dense') VBLL model - currently only running with one element in list
    train_vbll_diagonal = False
    train_vbll_lowrank = False
    train_vbll_dense_precision = False

    train_vbll_e = False

    train_pnn = True
    train_pe = False

    train_post_train = False
    train_recursive = False
    train_vbll_sngp = False

    compare_times = True
    different_dataset_sizes = [256]

class cfg_sub_test:
    def __init__(self, show_model, show_members):
        self.show_model = show_model
        self.show_members = show_members