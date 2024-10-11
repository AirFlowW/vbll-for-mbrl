class cfg_test:
    show_dataset = False
    show_mlp = False
    show_vbll_kl = False
    show_vbll_e = False
    show_vbll_e_members = False
    show_vbll_sngp = False
    show_pnn = False
    show_pe = False
    show_pe_members = False
    show_post_train = False

    train_mlp = True
    train_vbll_kl = True
    vbll_kl_weight = [1]#,10,50] # 1 for normal VBLL model currently only running with one element in list
    train_post_train = True
    train_vbll_e = False
    train_pnn = False
    train_pe = False
    train_vbll_sngp = False

    compare_times = True
    different_dataset_sizes = [64, 128, 256]#[32, 64]#, 128, 256, 512]#, 1024]
    # different_dataset_sizes = [32, 64]

class cfg_sub_test:
    def __init__(self, show_model, show_members):
        self.show_model = show_model
        self.show_members = show_members