class model_run_config:
    def __init__(self, model_name, train_cfg, models, train_fn, viz_model_fn, cfg_sub_t):
        self.model_name = model_name
        self.train_cfg = train_cfg
        self.models = models
        self.train_fn = train_fn
        self.viz_model_fn = viz_model_fn
        self.cfg_sub_t = cfg_sub_t