# AR-NCO

**When to Fuse? Exploring Cross-Depth Attention in Neural Combinatorial Optimization**

## Overview

AR-NCO investigates two cross-depth attention fusion strategies for neural combinatorial optimization. Full AR performs repeated fine-grained fusion during encoding, while Post AR performs a single fusion after encoding. This repository contains the retained AM, POMO, and preference optimization (PO) implementations for TSP and CVRP, saved evaluation instances, measured results, and trained model weights.

## Full AR and Post AR

- **Full AR:** attention fusion at successive residual sites, with the initial embedding and raw MHA/FFN sublayer outputs as historical sources.
- **Post AR:** one learned attention fusion over the initial embedding and encoder-layer outputs after encoding.

Model implementations are retained in `code/AM/nets/` and `code/POMO_PO/POMO/{TSP,CVRP}/POMO/`. PO uses the same routing policy family with a preference optimization objective. The checkpoint manifest specifies each historical model definition, including the TSP Full AR variant without FinalNorm.

## Repository Structure

```text
code/                 AM and POMO/PO models, environments, training code
configs/models/       Model construction settings for each checkpoint
configs/history/      Recorded training settings, with local paths omitted
configs/generalization/  Fixed zero-shot model selection and protocol
checkpoints/          Model tensor exports and their manifest
code/*/data/          Saved validation tensors (runtime-compatible locations)
datasets/             Dataset manifest, saved generalization tensors, LKH references
results/              Recorded main results, training observations, attention, generalization
scripts/              Strict evaluation, verification, generalization and plotting
figures/              Plots regenerated from saved attention statistics
docs/                 Reproducibility and third-party notices
```

## Requirements

Python 3.10 (validated), PyTorch, NumPy, SciPy, Matplotlib, tqdm, and pytz. GPU execution is optional; a CUDA GPU is recommended for the full 500-start TSP500 evaluation. `docs/tested_environment.json` records the original validation environment; `docs/tested_environment_cpu.json` records a separately installed CPU environment. Git LFS is recommended if tensor files are subsequently committed to GitHub; alternatively, distribute the companion archives as release assets.

## Installation

Extract `AR-NCO-GitHub.zip`, `AR-NCO-Data.zip` and `AR-NCO-Checkpoints.zip` into the same parent directory. All three contain an `AR-NCO/` folder and have disjoint file lists. From that folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_artifacts.py --output outputs/integrity.json
```

Install the PyTorch build appropriate for your CPU or CUDA environment. For the independently tested CPU environment, use `python -m pip install -r docs/requirements-cpu-lock.txt`. The provided companion archives contain actual tensor files. If obtaining tensors through a Git LFS repository instead, run `git lfs install` and `git lfs pull`. The source ZIP contains manifests and summaries; the data and checkpoint ZIPs supply their referenced artifacts.

## Data Preparation

The original saved data are included; generation is unnecessary for reproducing evaluation. `datasets/manifest.json` lists file names, shapes, instance counts, sizes, recorded seeds, and SHA256 values. Each validation tensor contains 10,000 instances. Generalization sets contain 1,000 instances per size, plus the original 10-instance diagnostic sets. AM and POMO validation copies are byte-identical for each task/size.

To generate **new** data in a separate location using the retained generator:

```bash
python code/AM/generate_data.py --problem tsp --graph_sizes 20 --dataset_size 10 --seed 1234 --name example --data_dir outputs/new_data
```

The original main-experiment TSP references are in `datasets/references/TSP/`. Completed generalization references, with continuous-coordinate lengths and tours, are in `datasets/references/generalization/`. LKH-3 is an external dependency; its source and executable are not included. See `docs/reproducibility.md` for the reference protocol.

## Training

Use `baseline`, `post_ar`, or `full_ar` for POMO/PO; use `attention`, `post_ar`, or `full_ar` for AM. The following are new training invocations, not replacements for historical configurations:

```bash
python code/POMO_PO/run.py --problem tsp --model post_ar --objective pomo --size 100 --device cpu --epochs 500 --episodes 100000 --batch-size 64 --dry-run
python code/POMO_PO/run.py --problem cvrp --model full_ar --objective po --size 50 --device cpu --dry-run
python code/AM/run.py --help
```

Remove `--dry-run` to train POMO/PO. For AM, for example:

```bash
python code/AM/run.py --problem tsp --model full_ar --graph_size 100 --n_epochs 500 --epoch_size 100000 --batch_size 64 --lr_model 0.0001 --baseline rollout --no_cuda --output_dir ../../outputs/am
```

Recorded configurations are retained separately in `configs/history/` and indexed in `checkpoints/manifest.json`. Weights-only exports support inference and initialization; they do not contain optimizer or RNG state for exact training continuation.

## Evaluation

List available models and strictly load every checkpoint:

```bash
python scripts/check_checkpoints.py --output outputs/checkpoint_validation.json
```

POMO/TSP100, using the exact epoch-500 baseline selected for generalization:

```bash
python scripts/evaluate.py --id POMO_TSP100_Baseline_Epoch500 --dataset code/POMO_PO/POMO/data/validation/tsp/uniform_size_100_seed_12345_tsp.pt --instances 10 --batch-size 2 --device cpu --output outputs/pomo_tsp100
```

CVRP example:

```bash
python scripts/evaluate.py --id POMO_CVRP20_Full_AR --dataset code/POMO_PO/POMO/data/validation/cvrp/uniform_size_20_seed_12345_cvrp.pt --instances 2 --batch-size 2 --device cpu --output outputs/pomo_cvrp20
```

Use any ID from `checkpoints/manifest.json` for AM, POMO, or PO. Evaluation uses strict state loading, greedy decoding, evaluation/inference mode, and augmentation factor 1. POMO/PO use one start per customer/city. Outputs contain tours, per-instance lengths, input hashes and a summary. Existing completed batches are reused only when the recorded protocol matches. Use a new output directory for a changed protocol. Tour validity, CVRP capacity and unchanged model parameters are checked.

## Zero-Shot Generalization

Three TSP100 epoch-500 policies are fixed in `configs/generalization/models.json`. They are evaluated on the same saved TSP100/200/500 instances, with POMO size equal to the test size, greedy decoding, augmentation factor 1, and no fine-tuning or search.

```bash
# Small diagnostic run using the first two saved formal instances per size
python scripts/generalization.py --instances 2 --batch-size 1 --device cpu --output outputs/generalization_smoke

# Full evaluation, with checkpointed output batches
python scripts/generalization.py --instances 1000 --batch-size 1 --device cuda:0 --output outputs/generalization
```

Choose a batch size that fits available memory; reduce it if needed without changing decoding. Reuse the same batch size across methods. Do not alter the shared test set. The included `results/generalization/all_per_instance.csv` preserves the original 9,000 measured neural lengths, shared LKH lengths, gaps, and original checkpoint hashes. Exported checkpoint hashes differ because optimizer and training state were removed; their model tensors are identical, with provenance recorded in the manifest.

## Reproducing Paper Results

| Content | Retained evidence / entry point |
|---|---|
| Table III: TSP | `results/TSP/paper_observations.csv` |
| Table IV: CVRP | `results/CVRP/paper_observations.csv` |
| Table V: ablation | `results/ablation/README.md` |
| Table VI: overhead | `results/overhead/README.md` |
| Table VII: generalization | `results/generalization/quality_table.csv`, `all_per_instance.csv` |
| Figures 3–4: attention | `results/attention/`, `scripts/plot_attention.py` |

Replot the saved attention statistics (without rerunning models):

```bash
python scripts/plot_attention.py --mode combined --post_ar_dir results/attention/PO_TSP100_Post_AR --full_ar_dir results/attention/POMO_TSP100_Full_AR --output_dir outputs/attention_combined
python scripts/plot_attention.py --mode full_ar_13source --data_dir results/attention/POMO_TSP100_Full_AR --output_dir outputs/attention_sources
```

Saved training curves can also be plotted without accessing local logs:

```bash
python scripts/plot_learning_curves.py --groups AM_TSP50_Baseline AM_TSP50_Post_AR AM_TSP50_Full_AR --output outputs/learning_curves.pdf
```

Raw per-instance attention arrays and standard deviations are included. The source weights round to M2 = 0.762, F2 = 0.012 and grouped L2 = 0.774. Model-based extraction and checkpoint associations are documented in `docs/reproducibility.md`.

## Checkpoints

`checkpoints/manifest.json` lists 59 saved model exports, their task and training size, historical model module, configuration, epoch indexing, source/export SHA256 and tensor-equivalence checks. These include retained training-chain finals and the exact epoch-500 generalization baseline. A chain's last checkpoint is not automatically the checkpoint selected for a paper table. See the recorded source observations for the supported associations.

## References

- [Attention, Learn to Solve Routing Problems! (AM)](https://github.com/wouterkool/attention-learn-to-route)
- [POMO: Policy Optimization with Multiple Optima for Reinforcement Learning](https://github.com/yd-kwon/POMO)
- [Preference Optimization for Combinatorial Optimization Problems](https://github.com/MingjunPan/PO4COPs)
- [LKH-3](http://akira.ruc.dk/~keld/research/LKH-3/)

Code licensing and third-party attribution are specified in `LICENSE` and `docs/third_party.md`. Research datasets, model weights and measured results are licensed under CC BY 4.0; see `DATA_LICENSE`.
