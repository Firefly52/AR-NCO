# Reproducibility

## Artifacts and provenance

The release separates code, saved datasets, measured results and model weights. Original saved tensors and numeric results are copied byte for byte. Checkpoints are explicitly marked weights-only exports: each model tensor is compared with its source tensor; optimizer, random-generator and pickled rollout-baseline objects are omitted. No model parameters are trained or changed during packaging.

The checkpoint manifest records historical model modules and construction settings. AM epochs follow the saved zero-based filenames; POMO/PO use their recorded epoch fields. Training-chain finals are supplied as available artifacts rather than relabeled as paper-selected checkpoints. History configurations retain recorded optimization values; unrelated local paths are omitted. They are provenance records, not directly executable command-line configuration files.

## Recorded model associations

- Generalization baseline: `POMO_TSP100_Baseline_Epoch500` (the chain's final epoch 510 is a separate artifact).
- Generalization Post AR: `POMO_TSP100_Post_AR_Run02`, nominal epoch 500. Recorded history: PO initialization through epoch 188, followed by POMO/RL through epoch 500.
- Generalization Full AR: `POMO_TSP100_Full_AR_Run01`, epoch 500, historical encoder without FinalNorm.
- Saved Post AR attention arrays: `PO_TSP100_Post_AR`, epoch 500, as identified by the original analysis configuration and tensor equality.
- Saved Full AR attention arrays: `POMO_TSP100_Full_AR_Run01`, epoch 500.

The attention and generalization Post AR weights are distinct artifacts. Their original analysis metadata and checkpoint hashes are retained; they are not interchanged.

## Main evaluation datasets

TSP tensors have shape `[10000, N, 2]`. CVRP tensors have shape `[10000, N+1, 3]`: row zero is the depot; the first two channels are coordinates; the last channel is normalized demand. AM and POMO copies share the same contents. The recorded seed is 12345.

Main TSP LKH reference files are retained with their original fields and values. A corresponding complete CVRP LKH reference-length collection was not found. Reference files are not automatically applied to other test sets.

## Generalization

The original saved sets use master seed 20261106 and stream seed `master + N` for formal data, plus 1000000 for diagnostic data. These files are copied, not regenerated. Reference records contain status, solver return code, instance index, length, tour and dataset hash.

Recorded LKH-3 settings: version 3.0.13, RUNS=10, MAX_TRIALS=1000, TIME_LIMIT=60 seconds per run, SEED=20261106, integer-coordinate scale 1000000. The global-best tour is captured using `TOUR_FILE`; lengths are recomputed on original floating-point coordinates. Gap means the per-instance relative gap to LKH-3, not a certified optimality gap. No bootstrap is used.

To rerun references after obtaining LKH-3 separately:

```bash
python scripts/lkh_reference.py --binary /path/to/externally/installed/LKH --dataset datasets/generalization/formal/tsp100.pt --instances 1000 --output outputs/lkh_tsp100
```

The packaged reference wrapper reuses the retained TSPLIB exporter and parameter builder. Its output is separate from saved results. No LKH reference computation is required to inspect or use the included reference data.

To extract attention on a small subset:

```bash
python scripts/analyze_attention_weights.py --model_type post_ar --checkpoint checkpoints/PO/TSP100/PO_TSP100_Post_AR/weights.pt --test_data code/POMO_PO/POMO/data/validation/tsp/uniform_size_100_seed_12345_tsp.pt --output_dir outputs/attention_post_smoke --num_instances 2 --batch_size 2 --device cpu --save_raw --verify
python scripts/analyze_attention_weights.py --model_type full_ar --checkpoint checkpoints/POMO/TSP100/POMO_TSP100_Full_AR_Run01/weights.pt --test_data code/POMO_PO/POMO/data/validation/tsp/uniform_size_100_seed_12345_tsp.pt --output_dir outputs/attention_full_smoke --num_instances 2 --batch_size 2 --device cpu --save_raw --verify
```

Remove `--num_instances 2` to process the original 10,000-instance dataset. Plotting the saved arrays needs no model execution.

## Validation scope

All released checkpoints are strictly loaded. Smoke tests cover AM/POMO/PO × baseline/Post AR/Full AR × TSP/CVRP, plus all nine model/size generalization combinations. Small tests check operational reproducibility, not a rerun of all paper experiments. Full training is not repeated. Source gaps remain explicitly identified in the associated result directories.

## Repository access and research-data deposit

A private repository cannot be accessed anonymously. Public visibility requires the repository owner's approval. GitHub account and commit metadata can identify authors. GitHub hosting alone does not establish a DOI or guarantee that a journal's data-repository requirements are met; check any platform or persistent-identifier requirement separately.
