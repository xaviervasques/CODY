# Task-based classification (symptom vs control)

This folder contains a **task-level** binary classification pipeline (e.g., Rest / Posture / Action segments).  
It evaluates how well different models discriminate symptomatic subjects from controls within task-specific segments.

## Inputs

By default, `task.py` loads `.xlsx` files from:

- `../datasets/rest_posture_action/` (relative to this folder)

Expected per-subject file structure:
- `From`, `To` columns delimiting task windows/segments
- one or more binary symptom columns
- distance columns (typically `*_distance`)
- control subjects identified by filenames starting with `C`

## Run

```bash
python task.py
```

## Dependencies

See the repository-level `requirements.txt`.
