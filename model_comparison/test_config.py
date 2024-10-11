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

    train_mlp = True
    train_vbll_kl = False
    vbll_kl_weight = [1]#,10,50] # 1 for normal VBLL model
    train_vbll_e = True
    train_pnn = False
    train_pe = True
    train_vbll_sngp = False

    compare_times = True
    different_dataset_sizes = [64, 128, 256]#[32, 64]#, 128, 256, 512]#, 1024]
    # different_dataset_sizes = [32, 64]

class cfg_sub_test:
    def __init__(self, show_model, show_members):
        self.show_model = show_model
        self.show_members = show_members