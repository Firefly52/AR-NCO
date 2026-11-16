#!/usr/bin/env python

import os
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from project_paths import resolve_path

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import glob
import json
import logging
import pprint as pp

import torch
import torch.optim as optim

from options import get_options
from train import train_epoch, validate, get_inner_model
from reinforce_baselines import NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline
from nets.attention_model import AttentionModel
from nets.attention_model_post_ar import AttentionModel as AttentionModelPostAR
from nets.attention_model_full_ar import AttentionModel as AttentionModelFullAR
from utils import torch_load_cpu, load_problem


##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 1


##########################################################################################
# Parameters

run_params = {
    # Data
    'problem': 'tsp',
    'graph_size': 50,
    'batch_size': 500,
    'epoch_size': 100000,
    'val_size': 10000,
    'val_dataset': 'auto',
    'validation_dataset_dirs': ['data/tsp', 'data/validation/tsp'],

    # Model
    'model': 'full_ar',
    'embedding_dim': 128,
    'hidden_dim': 128,
    'n_encode_layers': 6,
    'tanh_clipping': 10.,
    'normalization': 'batch',

    # Training
    'lr_model': 1e-4,
    'lr_critic': 1e-4,
    'lr_decay': 1.0,
    'eval_only': False,
    'n_epochs': 500,
    'seed': 12345,
    'max_grad_norm': 1.0,
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'exp_beta': 0.8,
    'baseline': 'rollout',
    'bl_alpha': 0.05,
    'bl_warmup_epochs': None,
    'eval_batch_size': 1024,
    'num_workers': 4,
    'checkpoint_encoder': False,
    'shrink_size': None,
    'data_distribution': None,

    # Misc
    'log_step': 50,
    'log_dir': str(PROJECT_ROOT / 'experiments/new_runs/AM_tensorboard'),
    'run_name': 'run',
    'output_dir': str(PROJECT_ROOT / 'experiments/new_runs/AM'),
    'epoch_start': 0,
    'checkpoint_epochs': 50,
    'load_path': None,
    'resume': None,
    'auto_resume_by_model': False,
    'attention_resume_path': 'outputs/tsp_50/tsp50_rollout_20260404T140406/epoch-54.pt',
    'post_ar_resume_path': 'outputs/tsp_50/tsp50_rollout_ar_20260404T143607/epoch-53.pt',
    'full_ar_resume_path': None,
    'no_tensorboard': False,
    'no_progress_bar': True,

    # Script-only controls
    'append_timestamp': True,
    'use_data_parallel': False,
    'data_parallel_device_ids': None,
}

logger_params = {
    'log_file': {
        'filename': 'log.txt',
    },
    'redirect_stdout_stderr': True,
}


if DEBUG_MODE:
    run_params.update({
        'n_epochs': 2,
        'epoch_size': 1024,
        'batch_size': 128,
        'val_size': 128,
        'eval_batch_size': 128,
        'no_tensorboard': True,
        'checkpoint_epochs': 0,
    })


##########################################################################################
# Logging

ORIGINAL_STDOUT = sys.stdout
ORIGINAL_STDERR = sys.stderr


class StreamToLogger(object):

    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.buffer = ''

    def write(self, message):
        if not message:
            return

        self.buffer += message
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            line = line.rstrip()
            if line:
                self.logger.log(self.level, line)

    def flush(self):
        if self.buffer.strip():
            self.logger.log(self.level, self.buffer.strip())
        self.buffer = ''

    def isatty(self):
        return False


def create_logger(opts):
    os.makedirs(opts.save_dir, exist_ok=True)

    log_filename = logger_params['log_file']['filename']
    log_filepath = os.path.join(opts.save_dir, log_filename)
    logger_params['log_file']['filepath'] = opts.save_dir

    formatter = logging.Formatter(
        '[%(asctime)s] %(filename)s(%(lineno)d) : %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    file_handler = logging.FileHandler(log_filepath, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(ORIGINAL_STDOUT)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    if logger_params.get('redirect_stdout_stderr', False):
        sys.stdout = StreamToLogger(root_logger, logging.INFO)
        sys.stderr = StreamToLogger(root_logger, logging.ERROR)

    logging.getLogger(__name__).info('Log file: %s', log_filepath)


def _print_config(opts, configured_run_params):
    logger = logging.getLogger(__name__)
    logger.info('DEBUG_MODE: %s', DEBUG_MODE)
    logger.info('USE_CUDA: %s, CUDA_DEVICE_NUM: %s', opts.use_cuda, opts.cuda_device_num)
    logger.info('run_params%s', configured_run_params)
    logger.info('logger_params%s', logger_params)
    logger.info('resolved_options%s', vars(opts))


def _resolve_auto_resume_path(opts):
    if not opts.auto_resume_by_model:
        return

    # Respect explicit resume/load_path settings from script config.
    if opts.resume is not None or opts.load_path is not None:
        return

    auto_resume_map = {
        'attention': opts.attention_resume_path,
        'post_ar': opts.post_ar_resume_path,
        'full_ar': opts.full_ar_resume_path,
    }

    resume_path = auto_resume_map.get(opts.model)
    assert resume_path is not None, "No auto resume checkpoint configured for model: {}".format(opts.model)
    assert os.path.isfile(resume_path), "Auto resume checkpoint not found: {}".format(resume_path)

    opts.resume = resume_path
    logging.getLogger(__name__).info('  [*] Auto resume enabled, using %s', opts.resume)


def _resolve_validation_dataset_path(opts):
    if opts.val_dataset is None:
        return None

    if opts.val_dataset != 'auto':
        assert os.path.isfile(opts.val_dataset), "Validation dataset not found: {}".format(opts.val_dataset)
        return opts.val_dataset

    # Map problem to its data directory so auto-resolution works for all problem types
    _PROBLEM_DATA_DIRS = {
        'tsp': 'data/tsp',
        'cvrp': 'data/vrp',
        'sdvrp': 'data/vrp',
        'op': 'data/op',
        'pctsp': 'data/pctsp',
    }

    dataset_dirs = list(opts.validation_dataset_dirs)
    if isinstance(dataset_dirs, str):
        dataset_dirs = [dataset_dirs]

    # Prepend the problem-specific data directory so it's searched first
    problem_dir = _PROBLEM_DATA_DIRS.get(opts.problem)
    if problem_dir is not None and problem_dir not in dataset_dirs:
        dataset_dirs.insert(0, problem_dir)

    search_patterns = [
        '{problem}{graph_size}_validation_seed*.pt',
        '{problem}{graph_size}_validation_seed*.pkl',
        '*validation*{graph_size}*.pt',
        '*validation*{graph_size}*.pkl',
        '*size_{graph_size}*{problem}.pt',
        '*{problem}*{graph_size}*.pt',
        '*{graph_size}*{problem}.pt',
    ]

    candidates = []
    for dataset_dir in dataset_dirs:
        dataset_dir = dataset_dir.format(problem=opts.problem, graph_size=opts.graph_size)
        for pattern in search_patterns:
            pattern = pattern.format(problem=opts.problem, graph_size=opts.graph_size)
            candidates.extend(glob.glob(os.path.join(dataset_dir, pattern)))

    candidates = sorted(set(candidates))
    if not candidates:
        raise FileNotFoundError(
            "No validation dataset found for problem={}, graph_size={} in {}".format(
                opts.problem, opts.graph_size, dataset_dirs
            )
        )

    def candidate_key(path):
        basename = os.path.basename(path)
        ext_priority = 0 if path.endswith('.pt') else 1
        validation_priority = 0 if 'validation' in basename else 1
        return ext_priority, validation_priority, basename

    return sorted(candidates, key=candidate_key)[0]


def run(opts, configured_run_params=None):
    if configured_run_params is None:
        configured_run_params = run_params

    create_logger(opts)
    _print_config(opts, configured_run_params)

    _resolve_auto_resume_path(opts)

    logger = logging.getLogger(__name__)

    # Pretty print the resolved run args
    logger.info('Resolved run options:\n%s', pp.pformat(vars(opts)))

    # Set the random seed
    torch.manual_seed(opts.seed)

    # Optionally configure tensorboard
    tb_logger = None
    if not opts.no_tensorboard:
        from tensorboard_logger import Logger as TbLogger
        tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, opts.graph_size), opts.run_name))

    os.makedirs(opts.save_dir, exist_ok=True)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts.save_dir, "args.json"), 'w', encoding='utf-8') as f:
        json.dump(vars(opts), f, indent=True)

    # Set the device
    if opts.use_cuda:
        cuda_device_count = torch.cuda.device_count()
        if opts.use_data_parallel:
            device_ids = opts.data_parallel_device_ids or list(range(cuda_device_count))
            assert device_ids, "data_parallel_device_ids cannot be empty"
            assert max(device_ids) < cuda_device_count, \
                "CUDA device_ids {} requested, but only {} CUDA device(s) are visible".format(
                    device_ids, cuda_device_count
                )
            opts.device = torch.device("cuda:{}".format(device_ids[0]))
        else:
            assert cuda_device_count > opts.cuda_device_num, \
                "CUDA device {} requested, but only {} CUDA device(s) are visible".format(
                    opts.cuda_device_num, cuda_device_count
                )
            torch.cuda.set_device(opts.cuda_device_num)
            opts.device = torch.device("cuda:{}".format(opts.cuda_device_num))
    else:
        opts.device = torch.device("cpu")

    # Figure out what's the problem
    problem = load_problem(opts.problem)

    # Load data from load_path
    load_data = {}
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        load_path = str(resolve_path(load_path, Path(__file__).parent))
        logger.info('  [*] Loading data from %s', load_path)
        load_data = torch_load_cpu(load_path)

    # Initialize model
    model_class = {
        'attention': AttentionModel,
        'post_ar': AttentionModelPostAR,
        'full_ar': AttentionModelFullAR,
    }.get(opts.model, None)
    assert model_class is not None, "Unknown model: {}".format(model_class)
    model = model_class(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
        shrink_size=opts.shrink_size
    ).to(opts.device)

    if opts.use_cuda and opts.use_data_parallel:
        device_ids = opts.data_parallel_device_ids or list(range(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    if opts.load_path or opts.resume:
        model_.load_state_dict(load_data['model'], strict=True)

    # Initialize baseline
    if opts.baseline == 'exponential':
        baseline = ExponentialBaseline(opts.exp_beta)
    elif opts.baseline in ('critic', 'critic_lstm'):
        raise ValueError('Critic experiments are preserved in archive; the paper uses rollout.')
    elif opts.baseline == 'rollout':
        baseline = RolloutBaseline(model, problem, opts)
    else:
        assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
        baseline = NoBaseline()

    if opts.bl_warmup_epochs > 0:
        baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)

    # Load baseline from data, make sure script is called with same type of baseline
    if 'baseline' in load_data:
        baseline.load_state_dict(load_data['baseline'])

    # Initialize optimizer
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': opts.lr_model}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )

    # Load optimizer state
    if 'optimizer' in load_data:
        optimizer.load_state_dict(load_data['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                # if isinstance(v, torch.Tensor):
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)

    # Initialize learning rate scheduler, decay by lr_decay once per epoch!
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)

    # Start the actual training loop
    opts.val_dataset = _resolve_validation_dataset_path(opts)
    if opts.val_dataset is None:
        logger.info(
            "Validation dataset: generated on the fly, size=%s, graph_size=%s",
            opts.val_size,
            opts.graph_size
        )
    else:
        logger.info("Validation dataset: %s", opts.val_dataset)

    val_dataset = problem.make_dataset(
        size=opts.graph_size, num_samples=opts.val_size, filename=opts.val_dataset, distribution=opts.data_distribution)

    if opts.resume:
        epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])

        torch.set_rng_state(load_data['rng_state'])
        if opts.use_cuda and 'cuda_rng_state' in load_data:
            saved_cuda_rng = load_data['cuda_rng_state']
            if isinstance(saved_cuda_rng, (list, tuple)):
                available_cuda_devices = torch.cuda.device_count()
                restore_count = min(len(saved_cuda_rng), available_cuda_devices)
                for device_idx in range(restore_count):
                    torch.cuda.set_rng_state(saved_cuda_rng[device_idx], device=device_idx)
                if restore_count != len(saved_cuda_rng):
                    logger.info(
                        "  [!] Resume RNG state mismatch: saved %s CUDA device states, restored %s on current run",
                        len(saved_cuda_rng), restore_count
                    )
            else:
                torch.cuda.set_rng_state(saved_cuda_rng)
        # Set the random states
        # Dumping of state was done before epoch callback, so do that now (model is loaded)
        baseline.epoch_callback(model, epoch_resume)
        logger.info("Resuming after %s", epoch_resume)
        opts.epoch_start = epoch_resume + 1

    if opts.eval_only:
        validate(model, val_dataset, opts)
    else:
        for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
            train_epoch(
                model,
                optimizer,
                baseline,
                lr_scheduler,
                epoch,
                val_dataset,
                problem,
                tb_logger,
                opts
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Attention Model for Vehicle Routing Problems')
    parser.add_argument('--problem', type=str, default=None,
                        choices=['tsp', 'cvrp'],
                        help='Problem type (default: use run_params value)')
    parser.add_argument('--model', type=str, default=None,
                        choices=['attention', 'post_ar', 'full_ar'],
                        help='Model type (default: use run_params value)')
    parser.add_argument('--graph_size', type=int, default=None,
                        help='Number of nodes (default: use run_params value)')
    parser.add_argument('--n_epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--checkpoint_epochs', type=int, default=None,
                        help='Save checkpoint every N epochs (0 = never)')
    parser.add_argument('--val_dataset', type=str, default=None,
                        help='Path to validation dataset .pt/.pkl file, "auto", or "None" to generate on-the-fly')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Training batch size')
    parser.add_argument('--epoch_size', type=int, default=None,
                        help='Number of training samples per epoch')
    parser.add_argument('--lr_model', type=float, default=None,
                        help='Learning rate for the model')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--baseline', type=str, default=None,
                        choices=['rollout', 'exponential', 'none'],
                        help='Baseline type')
    parser.add_argument('--cuda_device_num', type=int, default=None,
                        help='CUDA device number')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Disable CUDA')
    parser.add_argument('--eval_only', action='store_true', default=None,
                        help='Only evaluate, no training')
    parser.add_argument('--load_path', type=str, default=None,
                        help='Path to load model weights from')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to resume training from checkpoint')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Run name for output directory')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory')

    args = parser.parse_args()

    # Override run_params with command-line arguments (only non-None values)
    for key, value in vars(args).items():
        if value is not None:
            if key == 'no_cuda' and value:
                run_params['use_cuda'] = False
            elif key == 'baseline' and value == 'none':
                run_params['baseline'] = None
            elif key == 'val_dataset':
                if value.lower() == 'none':
                    run_params['val_dataset'] = None
                elif value.lower() == 'auto':
                    run_params['val_dataset'] = 'auto'
                else:
                    run_params['val_dataset'] = value
            elif key not in ('no_cuda',):
                run_params[key] = value

    # Auto-generate run_name from problem + graph_size + model if not explicitly set
    if args.run_name is None:
        run_params['run_name'] = "{}{}_{}".format(
            run_params['problem'], run_params['graph_size'], run_params['model']
        )

    # Print final config for confirmation
    import pprint
    print("=" * 60)
    print("Running experiment with the following config:")
    print("-" * 60)
    for k, v in run_params.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    try:
        run(get_options(run_params), run_params)
    except Exception:
        logging.getLogger(__name__).exception("Unhandled exception")
        raise
