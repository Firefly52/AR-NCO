# Download and archive layout

Download these three archives and extract them into the same directory:

1. `AR-NCO-GitHub.zip`: source code, documentation, model/dataset manifests, historical settings, result summaries and compact figure previews.
2. `AR-NCO-Data.zip`: saved evaluation tensors, LKH reference records, raw attention arrays and saved neural tours.
3. `AR-NCO-Checkpoints.zip`: 59 model tensor exports (no optimizer state).

All archives use the same top-level folder, `AR-NCO/`, with disjoint file lists. The resulting folder is the complete project. Do not place one extracted `AR-NCO/` directory inside another. From the merged folder, run:

```bash
python scripts/verify_artifacts.py --output outputs/integrity.json
python scripts/check_checkpoints.py --output outputs/checkpoints.json
```

`docs/file_manifest.json` assigns every distributed file to an archive and provides a SHA256 checksum. `SHA256SUMS.txt`, distributed alongside the ZIPs, checks the archives themselves. Internal audit reports, installation environments and smoke outputs are not part of the public archives.

For a later GitHub upload, keep source files in Git and place the two large companion archives in GitHub Releases, or use Git LFS for their extracted tensors. The `.gitattributes` file defines LFS patterns for PT/NPY/NPZ files. A code-only checkout requires the two companion archives before dataset/model evaluation. The ZIPs are prepared locally; this delivery does not create or push a GitHub repository.
