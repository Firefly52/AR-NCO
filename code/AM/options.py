import os
import time
from types import SimpleNamespace

import torch


DEFAULT_RUN_PARAMS = {
    # Data
    'problem': 'tsp',
    'graph_size': 20,
    'batch_size': 512,
    'epoch_size': 1280000,
    'val_size': 10000,
    'val_dataset': 'auto',
    'validation_dataset_dirs': ['data/tsp', 'data/validation/tsp'],

    # Model
    'model': 'attention',
    'embedding_dim': 128,
    'hidden_dim': 128,
    'n_encode_layers': 3,
    'tanh_clipping': 10.,
    'normalization': 'batch',

    # Training
    'lr_model': 1e-4,
    'lr_critic': 1e-4,
    'lr_decay': 1.0,
    'eval_only': False,
    'n_epochs': 100,
    'seed': 1234,
    'max_grad_norm': 1.0,
    'use_cuda': None,
    'cuda_device_num': 0,
    'exp_beta': 0.8,
    'baseline': None,
    'bl_alpha': 0.05,
    'bl_warmup_epochs': None,
    'eval_batch_size': 1024,
    'num_workers': 0,
    'checkpoint_encoder': False,
    'shrink_size': None,
    'data_distribution': None,

    # Misc
    'log_step': 50,
    'log_dir': 'logs',
    'run_name': 'run',
    'output_dir': 'outputs',
    'epoch_start': 0,
    'checkpoint_epochs': 1,
    'load_path': None,
    'resume': None,
    'auto_resume_by_model': False,
    'attention_resume_path': 'outputs/tsp_100/tsp100_rollout_20260404T140406/epoch-54.pt',
    'post_ar_resume_path': 'outputs/tsp_100/tsp100_rollout_ar_20260404T143607/epoch-53.pt',
    'full_ar_resume_path': None,
    'no_tensorboard': False,
    'no_progress_bar': False,

    # Script-only controls
    'append_timestamp': True,
    'use_data_parallel': False,
    'data_parallel_device_ids': None,
}


def get_options(run_params=None):
    params = DEFAULT_RUN_PARAMS.copy()
    if run_params is not None:
        unknown_keys = sorted(set(run_params) - set(params))
        assert not unknown_keys, "Unknown run parameter(s): {}".format(", ".join(unknown_keys))
        params.update(run_params)

    opts = SimpleNamespace(**params)

    if opts.use_cuda is None:
        opts.use_cuda = torch.cuda.is_available()
    else:
        opts.use_cuda = bool(opts.use_cuda) and torch.cuda.is_available()
    opts.no_cuda = not opts.use_cuda

    if opts.append_timestamp:
        opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))

    opts.save_dir = os.path.join(
        opts.output_dir,
        "{}_{}".format(opts.problem, opts.graph_size),
        opts.run_name
    )

    if opts.bl_warmup_epochs is None:
        opts.bl_warmup_epochs = 1 if opts.baseline == 'rollout' else 0

    assert opts.model in ('attention', 'post_ar', 'full_ar'), "Unknown model: {}".format(opts.model)
    assert (opts.bl_warmup_epochs == 0) or (opts.baseline == 'rollout')
    assert opts.epoch_size % opts.batch_size == 0, "Epoch size must be integer multiple of batch size!"
    return opts
