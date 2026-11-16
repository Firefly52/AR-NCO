# Release validation

Validated on 2026-11-16. No full training or replacement paper experiment was performed.

- 59 checkpoint exports load with `strict=True` in both the original environment and a separately installed PyTorch 2.5.1 CPU environment.
- Exported parameters are tensor-identical to their original checkpoints. Model definition files are byte-identical to the retained project implementations.
- 18 two-instance inference tests cover AM/POMO/PO × baseline/Post AR/Full AR × TSP20/CVRP20 in each environment.
- All nine zero-shot model/size combinations pass: two saved instances each on CUDA, one each in the clean CPU environment. Tours are valid and parameters unchanged.
- Dataset and checkpoint hashes match 77 manifest entries.
- All 9,000 saved generalization rows are checked against the saved shared references; all 3,000 LKH and 9,000 neural tours are valid and their lengths agree with the saved coordinates and measurements.
- Post AR and Full AR attention extraction pass two-instance recording/forward checks. Saved-statistics plots and a retained AM learning-curve plot render successfully.
- Training dry-run/help commands, the retained data generator, and reference-interface CLI are exercised. LKH-3 itself is not bundled or rerun during packaging.
- `pip check` succeeds in the clean environment. `docs/requirements-cpu-lock.txt` records its installed dependencies. The original environment has inconsistent package metadata, so its runtime module versions are recorded separately.
- The release copy changes path resolution, strict-loading safeguards and attention checkpoint metadata/selection only; model architectures and original numeric artifacts are preserved.

The accompanying private delivery report records source gaps and detailed test outputs. Read result-directory notes before interpreting the available artifacts as a reproduction of a particular table row.
