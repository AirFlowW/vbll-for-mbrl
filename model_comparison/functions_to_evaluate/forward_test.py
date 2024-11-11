def forward_pass(dataloader, model, test_cfg, torch_no_grad = True, verbose = True):
    """Evaluate forward pass of model on dataloader."""
    
    passes = 0
    for epoch in range(test_cfg.NUM_EPOCHS):
        if torch_no_grad:
            model.eval()
        for train_step, (x, y) in enumerate(dataloader):
            out = model(x)
            passes += 1

    print('Forward pass complete')
    print(f'Epochs: {test_cfg.NUM_EPOCHS}')
    print(f'Passes: {passes}')
    