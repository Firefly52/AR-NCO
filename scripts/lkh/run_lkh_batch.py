import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# LKH executable and workspace config.
# Priority:
# 1) env var LKH_EXECUTABLE
# 2) this default command name resolved via PATH
LKH_EXECUTABLE = 'LKH'
INSTANCES_DIR = 'POMO/TSP/lkh_workspace/tsp_instances'
PAR_DIR = 'POMO/TSP/lkh_workspace/par_files'
TOUR_DIR = 'POMO/TSP/lkh_workspace/tour_files'
LOG_DIR = 'POMO/TSP/lkh_workspace/lkh_logs'
META_JSON = 'POMO/TSP/lkh_workspace/instances_meta.json'
RUNS = 10
MAX_TRIALS = 10000
TIME_LIMIT = 60
DEFAULT_WORKERS = max(1, (os.cpu_count() or 1) - 1)
SKIP_EXISTING = True


def _build_par_content(problem_file, output_tour_file):
    lines = [
        f'PROBLEM_FILE = {problem_file}',
        f'OUTPUT_TOUR_FILE = {output_tour_file}',
        f'RUNS = {RUNS}',
        f'MAX_TRIALS = {MAX_TRIALS}',
        f'TIME_LIMIT = {TIME_LIMIT}',
        'TRACE_LEVEL = 1',
    ]
    return '\n'.join(lines) + '\n'


def _env_flag(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ('0', 'false', 'no', 'off')


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
        return parsed if parsed > 0 else default
    except ValueError:
        return default


def _resolve_lkh_executable():
    env_exec = os.environ.get('LKH_EXECUTABLE')
    candidate = env_exec if env_exec else LKH_EXECUTABLE

    # If an explicit absolute/relative path is provided, validate directly.
    if os.path.sep in candidate:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
        raise FileNotFoundError(
            f'LKH executable not found or not executable: {candidate}\n'
            'Please set a valid path in env var LKH_EXECUTABLE.'
        )

    resolved = shutil.which(candidate)
    if resolved:
        return resolved

    raise FileNotFoundError(
        'Cannot find LKH executable in PATH.\n'
        'Fix options:\n'
        '1) Install LKH and ensure command "LKH" is in PATH\n'
        '2) Run with env var, e.g.:\n'
        '   LKH_EXECUTABLE=/absolute/path/to/LKH python POMO/TSP/run_lkh_batch.py'
    )


def _resolve_meta_path(workspace_root):
    env_meta = os.environ.get('LKH_META_JSON')
    if env_meta:
        p = Path(env_meta)
        return p if p.is_absolute() else (workspace_root / p)
    return workspace_root / META_JSON


def _resolve_workspace_paths(workspace_root):
    meta_path = _resolve_meta_path(workspace_root)
    instances_dir = workspace_root / INSTANCES_DIR
    par_dir = workspace_root / PAR_DIR
    tour_dir = workspace_root / TOUR_DIR
    log_dir = workspace_root / LOG_DIR
    problem_size = None

    if meta_path.exists():
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        problem_size = meta.get('problem_size')
        rel = meta.get('workspace_rel', {}) if isinstance(meta, dict) else {}

        instances_dir = workspace_root / rel.get('instances_dir', INSTANCES_DIR)
        par_dir = workspace_root / rel.get('par_dir', PAR_DIR)
        tour_dir = workspace_root / rel.get('tour_dir', TOUR_DIR)
        log_dir = workspace_root / rel.get('log_dir', LOG_DIR)

    return {
        'meta_path': meta_path,
        'instances_dir': instances_dir,
        'par_dir': par_dir,
        'tour_dir': tour_dir,
        'log_dir': log_dir,
        'problem_size': problem_size,
    }


def _is_valid_tour_file(path):
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _process_one_instance(tsp_path, lkh_exec, par_dir, tour_dir, log_dir, skip_existing):
    stem = tsp_path.stem
    par_path = par_dir / f'{stem}.par'
    tour_path = tour_dir / f'{stem}.tour'
    log_path = log_dir / f'{stem}.log'

    if skip_existing and _is_valid_tour_file(tour_path):
        return {
            'name': stem,
            'status': 'skipped',
            'returncode': 0,
            'par': str(par_path),
            'log': str(log_path),
            'tour': str(tour_path),
        }

    par_content = _build_par_content(str(tsp_path), str(tour_path))
    with open(par_path, 'w', encoding='utf-8') as f:
        f.write(par_content)

    with open(log_path, 'w', encoding='utf-8') as log_file:
        result = subprocess.run([lkh_exec, str(par_path)], stdout=log_file, stderr=subprocess.STDOUT)

    if result.returncode == 0 and _is_valid_tour_file(tour_path):
        return {
            'name': stem,
            'status': 'success',
            'returncode': result.returncode,
            'par': str(par_path),
            'log': str(log_path),
            'tour': str(tour_path),
        }

    return {
        'name': stem,
        'status': 'failed',
        'returncode': result.returncode,
        'par': str(par_path),
        'log': str(log_path),
        'tour': str(tour_path),
    }


def main():
    lkh_exec = _resolve_lkh_executable()
    print(f'Using LKH executable: {lkh_exec}')

    workers = _env_int('LKH_WORKERS', DEFAULT_WORKERS)
    skip_existing = _env_flag('LKH_SKIP_EXISTING', SKIP_EXISTING)
    print(f'Parallel workers: {workers}')
    print(f'Skip existing tours: {skip_existing}')

    workspace_root = Path(__file__).resolve().parents[2]
    paths = _resolve_workspace_paths(workspace_root)
    instances_dir = paths['instances_dir']
    par_dir = paths['par_dir']
    tour_dir = paths['tour_dir']
    log_dir = paths['log_dir']
    problem_size = paths['problem_size']

    if paths['meta_path'].exists():
        print(f'Using meta file: {paths["meta_path"]}')
    print(f'Instances dir: {instances_dir}')
    print(f'Tour dir: {tour_dir}')

    par_dir.mkdir(parents=True, exist_ok=True)
    tour_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    tsp_files = sorted(instances_dir.glob('*.tsp'))
    if not tsp_files:
        raise FileNotFoundError(f'No .tsp files found in: {instances_dir}')

    summary = {'total': len(tsp_files), 'success': 0, 'skipped': 0, 'failed': [], 'workers': workers}

    completed = 0
    if workers == 1:
        for tsp_path in tsp_files:
            item = _process_one_instance(tsp_path, lkh_exec, par_dir, tour_dir, log_dir, skip_existing)
            completed += 1
            if item['status'] == 'success':
                summary['success'] += 1
            elif item['status'] == 'skipped':
                summary['skipped'] += 1
            else:
                summary['failed'].append(item)

            if completed % 100 == 0 or completed == len(tsp_files):
                print(f'Progress: {completed}/{len(tsp_files)}')
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_process_one_instance, tsp_path, lkh_exec, par_dir, tour_dir, log_dir, skip_existing)
                for tsp_path in tsp_files
            ]

            for future in as_completed(futures):
                item = future.result()
                completed += 1

                if item['status'] == 'success':
                    summary['success'] += 1
                elif item['status'] == 'skipped':
                    summary['skipped'] += 1
                else:
                    summary['failed'].append(item)

                if completed % 100 == 0 or completed == len(tsp_files):
                    print(f'Progress: {completed}/{len(tsp_files)}')

    if problem_size is None:
        summary_path = workspace_root / 'POMO/TSP/lkh_workspace/lkh_summary.json'
    else:
        summary_path = workspace_root / f'POMO/TSP/lkh_workspace/lkh_summary_size_{problem_size}.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(
        f"LKH finished: success={summary['success']}, skipped={summary['skipped']}, "
        f"failed={len(summary['failed'])}, total={summary['total']}"
    )
    print(f'Summary saved to: {summary_path}')
    if summary['failed']:
        print('Failed examples saved in summary file.')


if __name__ == '__main__':
    main()
