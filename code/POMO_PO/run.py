#!/usr/bin/env python3
"""Train/evaluate the paper's baseline, Post AR and Full AR implementations."""
import argparse
import copy
import importlib
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POMO = Path(__file__).resolve().parent / 'POMO'
sys.path.insert(0, str(ROOT))
from project_paths import resolve_path


def model_module(problem, model, checkpoint=None):
    prefix = problem.upper()
    module = prefix + {'baseline': 'Model', 'post_ar': 'ModelPostAR', 'full_ar': 'ModelFullAR'}[model]
    if problem == 'tsp' and model == 'full_ar' and checkpoint:
        import torch
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)['model_state_dict']
        if 'encoder.final_norm.weight' not in state:
            module = 'TSPModelFullARLegacy'
    return module


def configure_imports(problem):
    folder = POMO / problem.upper()
    for path in (POMO, folder, folder / 'POMO'):
        sys.path.insert(0, str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['train', 'eval'], default='train')
    parser.add_argument('--problem', choices=['tsp', 'cvrp'], default='tsp')
    parser.add_argument('--model', choices=['baseline', 'post_ar', 'full_ar'], default='post_ar')
    parser.add_argument('--objective', choices=['pomo', 'po'], default='pomo')
    parser.add_argument('--size', type=int, choices=[20, 50, 100], default=100)
    parser.add_argument('--device', default='cpu', help='cpu or cuda:N')
    parser.add_argument('--checkpoint', help='Current or historical checkpoint path')
    parser.add_argument('--config', help='Historical configs/paper_runs/run_NNN.json')
    parser.add_argument('--segment', type=int, default=-1, help='Historical configuration segment index')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--episodes', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--validation-episodes', type=int, default=10000)
    parser.add_argument('--output-dir')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    cfg = {}
    segment = {}
    if args.config:
        cfg = json.loads(resolve_path(args.config).read_text())
        if cfg.get('active_paper_run') is False:
            parser.error('This configuration is archived; see its archive_reason and the cleanup report.')
        if cfg['project'] != 'POMO_PO':
            parser.error('AM configurations use code/AM/run.py.')
        segment = copy.deepcopy(cfg['segments'][args.segment])
        args.problem, args.size = cfg['problem'], cfg['size']
        args.model = {'baseline_or_other': 'baseline'}.get(cfg['variant'], cfg['variant'])
        if args.model not in ('baseline', 'post_ar', 'full_ar'):
            parser.error('This archived configuration is not one of the paper models.')
        historical_module = segment.get('trainer_params', {}).get('model_module', '')
        if historical_module and historical_module not in {
            'TSPModel', 'CVRPModel', 'TSPModelARInitEmbed', 'CVRPModelARInitEmbed',
            'TSPModelFullAR', 'CVRPModelFullAR', 'TSPModelPostAR', 'CVRPModelPostAR'}:
            parser.error('This configuration uses an archived exploratory variant.')
    checkpoint = resolve_path(args.checkpoint) if args.checkpoint else None
    if checkpoint and not checkpoint.is_file():
        parser.error('Checkpoint does not exist: ' + str(checkpoint))
    if args.mode == 'eval' and not checkpoint:
        parser.error('--checkpoint is required for evaluation.')
    configure_imports(args.problem)
    module = model_module(args.problem, args.model, checkpoint)
    dataset = POMO / 'data/validation' / args.problem / f'uniform_size_{args.size}_seed_12345_{args.problem}.pt'
    env = {'problem_size': args.size, 'pomo_size': args.size}
    model = dict(embedding_dim=128, sqrt_embedding_dim=128 ** .5, encoder_layer_num=6,
                 decoder_layer_num=1, qkv_dim=16, head_num=8, logit_clipping=50,
                 ff_hidden_dim=512, eval_type='argmax')
    model.update(segment.get('model_params', {}))
    # Infer tensor dimensions for checkpoint loading, never normalization parameters.
    if checkpoint:
        import torch
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)['model_state_dict']
        layer_indices = {int(key.split('.')[2]) for key in state if key.startswith('encoder.layers.')}
        model['encoder_layer_num'] = max(layer_indices) + 1
    train = copy.deepcopy(segment.get('trainer_params', {}))
    loss = train.get('loss_type', 'po_loss' if args.objective == 'po' else 'rl_loss')
    if loss not in ('rl_loss', 'po_loss'):
        parser.error('Only the paper POMO and PO objectives are active.')
    train.update(use_cuda=args.device.startswith('cuda'),
                 cuda_device_num=int(args.device.split(':')[1]) if ':' in args.device else 0,
                 epochs=args.epochs if args.epochs is not None else train.get('epochs', 500),
                 train_episodes=args.episodes if args.episodes is not None else train.get('train_episodes', 100000),
                 train_batch_size=args.batch_size if args.batch_size is not None else train.get('train_batch_size', 64),
                 loss_type=loss, alpha=train.get('alpha', .05), local_search=False,
                 augmentation_enable=False, aug_factor=8, aug_batch_size=8,
                 optimizer_step_interval=1, model_module=module,
                 model_load={'enable': bool(checkpoint), 'checkpoint_path': str(checkpoint), 'load_optimizer': True},
                 epoch_test={'enable': True, 'validation_set_path': str(dataset),
                             'episodes': args.validation_episodes, 'batch_size': min(args.validation_episodes, 500)})
    styles = POMO / 'utils/log_image_style'
    train['logging'] = {'model_save_interval': 50, 'img_save_interval': 5000, 'tensorboard_enable': False,
                        'log_image_params_1': {'json_foldername': str(styles), 'filename': f'style_{args.problem}_{args.size}.json'},
                        'log_image_params_2': {'json_foldername': str(styles), 'filename': 'style_loss_1.json'}}
    optimizer = copy.deepcopy(segment.get('optimizer_params', {
        'optimizer': {'lr': 3e-4, 'weight_decay': 1e-6},
        'scheduler': {'milestones': [3001], 'gamma': .2}}))
    output = resolve_path(args.output_dir) if args.output_dir else ROOT / 'experiments/new_runs/POMO_PO' / (
        datetime.now().strftime('%Y%m%d_%H%M%S_%f') + f'_{args.problem}{args.size}_{args.model}_{loss}')
    summary = {'mode': args.mode, 'problem': args.problem, 'size': args.size, 'model': args.model,
               'model_module': module, 'checkpoint': str(checkpoint) if checkpoint else None,
               'output': str(output), 'env_params': env, 'model_params': model,
               'trainer_params': train, 'optimizer_params': optimizer}
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    import torch
    torch.set_num_threads(2)
    from utils.utils import create_logger, set_result_folder
    set_result_folder(str(output))
    create_logger(log_file={'filename': 'log.txt'})
    (output / 'configuration.json').write_text(json.dumps(summary, indent=2))
    if args.mode == 'train':
        trainer_cls = getattr(importlib.import_module(args.problem.upper() + 'Trainer'), args.problem.upper() + 'Trainer')
        trainer = trainer_cls(env_params=env, model_params=model, optimizer_params=optimizer, trainer_params=train)
        trainer.run()
    else:
        params = {'use_cuda': train['use_cuda'], 'cuda_device_num': train['cuda_device_num'],
                  'model_module': module, 'model_load': {'checkpoint_path': str(checkpoint)},
                  'test_episodes': args.episodes or 10000, 'test_batch_size': args.batch_size or 100,
                  'augmentation_enable': False, 'aug_factor': 8, 'beta': .05,
                  'gap_eval': {'enable': False}, 'test_data_load': {'enable': True, 'filename': str(dataset)}}
        env['validation_set_path'] = str(dataset)
        tester_cls = getattr(importlib.import_module(args.problem.upper() + 'Tester'), args.problem.upper() + 'Tester')
        tester_cls(env_params=env, model_params=model, tester_params=params).run()


if __name__ == '__main__':
    main()
