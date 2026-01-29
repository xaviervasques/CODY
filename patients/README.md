# Patient-based symptom classification

This folder contains a **patient-level** classification workflow.  
Compared to window-level approaches, it aggregates features across **all valid windows of a subject** and produces subject-level predictions.

## Inputs

By default, `patient.py` loads `.xlsx` files from:

- `../datasets/dataset_lc/` (relative to this folder)

Each subject file is expected to contain:
- `From`, `To` columns delimiting window boundaries
- one or more binary symptom columns (e.g. `Dystonia`, `Tremor`, …)
- pose-derived distance time series columns (typically `*_distance`)
- control subjects identified by filenames starting with `C`

## Run

```bash
python patient.py
```

Results are written to the local folder (see the script’s `output_dir` / default behavior).

## Dependencies

See the repository-level `requirements.txt`.
