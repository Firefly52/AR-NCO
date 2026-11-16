#!/usr/bin/env python3
"""Plot explicitly selected paper lineages without combining independent runs."""
import argparse
import json
from pathlib import Path
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from project_paths import resolve_path


def points(folder):
    import csv
    with (folder / 'learning_curve.csv').open() as f:
        values = [(int(x['epoch']), float(x['validation_length'])) for x in csv.DictReader(f)]
    metadata = json.loads((folder / 'curve_metadata.json').read_text())
    return values, metadata['conflicting_epochs']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--groups', nargs='+', required=True, help='Names in results/training/')
    parser.add_argument('--output', default=str(ROOT / 'outputs/learning_curves.pdf'))
    args = parser.parse_args()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in args.groups:
        folder = resolve_path('results/training/' + name)
        values, conflicts = points(folder)
        if not values:
            raise ValueError('No validation observations in ' + name)
        epochs, scores = zip(*values)
        # Conflicting values remain separate points, without inventing a single curve.
        if conflicts:
            ax.scatter(epochs, scores, s=8, label=name + ' (conflicting epochs)')
        else:
            ax.plot(epochs, scores, linewidth=1, label=name)
    ax.set(xlabel='Logged epoch (AM starts at 0; POMO/PO starts at 1)', ylabel='Validation route length')
    ax.legend(fontsize=7)
    ax.grid(alpha=.2)
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    print(output)


if __name__ == '__main__':
    main()
