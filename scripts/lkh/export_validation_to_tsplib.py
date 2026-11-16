import argparse
import json
from pathlib import Path

import torch


# Input validation set (.pt tensor/list), output TSPLIB files.
VALIDATION_PT = 'POMO/data/validation/tsp/uniform_size_20_seed_12345_tsp.pt'
LKH_WORKSPACE_DIR = 'POMO/TSP/lkh_workspace'
SCALE = 1000000


def _to_int_coord(x):
    return int(round(float(x) * SCALE))


def _write_tsplib_file(points, out_path, name):
    n = points.size(0)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f'NAME : {name}\n')
        f.write('TYPE : TSP\n')
        f.write(f'DIMENSION : {n}\n')
        f.write('EDGE_WEIGHT_TYPE : EUC_2D\n')
        f.write('NODE_COORD_SECTION\n')
        for i in range(n):
            x = _to_int_coord(points[i, 0])
            y = _to_int_coord(points[i, 1])
            f.write(f'{i + 1} {x} {y}\n')
        f.write('EOF\n')


def _relative_to_workspace(path_obj, workspace_root):
    try:
        return str(path_obj.resolve().relative_to(workspace_root.resolve()))
    except ValueError:
        return str(path_obj.resolve())


def main():
    parser = argparse.ArgumentParser(description='Export validation .pt to TSPLIB files for LKH')
    parser.add_argument(
        '--validation-pt',
        type=str,
        default=VALIDATION_PT,
        help='Path to validation .pt file (workspace-relative or absolute).',
    )
    args = parser.parse_args()

    workspace_root = Path(__file__).resolve().parents[2]
    raw_val_path = Path(args.validation_pt)
    val_path = raw_val_path if raw_val_path.is_absolute() else (workspace_root / raw_val_path)

    data = torch.load(val_path, map_location='cpu', weights_only=False)
    data = torch.as_tensor(data, dtype=torch.float)
    if data.dim() != 3 or data.size(-1) != 2:
        raise ValueError(f'Unexpected validation tensor shape: {tuple(data.shape)}')

    num_instances = data.size(0)
    problem_size = int(data.size(1))

    size_workspace = workspace_root / LKH_WORKSPACE_DIR / f'size_{problem_size}'
    output_dir = size_workspace / 'tsp_instances'
    par_dir = size_workspace / 'par_files'
    tour_dir = size_workspace / 'tour_files'
    log_dir = size_workspace / 'lkh_logs'
    meta_path = size_workspace / 'instances_meta.json'
    active_meta_path = workspace_root / LKH_WORKSPACE_DIR / 'instances_meta.json'

    output_dir.mkdir(parents=True, exist_ok=True)
    par_dir.mkdir(parents=True, exist_ok=True)
    tour_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        'validation_pt': str(val_path.resolve()),
        'num_instances': int(num_instances),
        'problem_size': problem_size,
        'scale': SCALE,
        'workspace_rel': {
            'instances_dir': _relative_to_workspace(output_dir, workspace_root),
            'par_dir': _relative_to_workspace(par_dir, workspace_root),
            'tour_dir': _relative_to_workspace(tour_dir, workspace_root),
            'log_dir': _relative_to_workspace(log_dir, workspace_root),
            'meta_json': _relative_to_workspace(meta_path, workspace_root),
        },
        'files': []
    }

    for idx in range(num_instances):
        name = f'tsp_{idx:06d}'
        tsp_path = output_dir / f'{name}.tsp'
        _write_tsplib_file(data[idx], tsp_path, name)
        meta['files'].append({'index': idx, 'tsp_file': str(tsp_path)})

    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    active_meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(active_meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print(f'Exported {num_instances} instances to: {output_dir}')
    print(f'Meta saved to: {meta_path}')
    print(f'Active meta updated: {active_meta_path}')


if __name__ == '__main__':
    main()
