# Window-based classification (symptom vs control)

This folder contains a **window-level** binary classification pipeline.  
For each symptom, it extracts statistical / spectral / complexity descriptors from pose-derived distance time series in each window and evaluates symptom-vs-control performance.

## Inputs

By default, `windows.py` loads `.xlsx` files from:

- `../datasets/dataset_lc/` (relative to this folder)

Each subject file is expected to contain:
- `From`, `To` window boundaries
- one or more binary symptom columns
- distance columns (typically `*_distance`)
- control subjects identified by filenames starting with `C`

## Run

```bash
python windows.py
```

Outputs (Excel workbooks) are written into this folder by default.

## Dependencies

See the repository-level `requirements.txt`.
