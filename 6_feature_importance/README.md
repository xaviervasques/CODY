# Feature Importance with LightGBM

This repository contains a Python script to compute and visualize feature importance using **LightGBM** and additional tools.
It supports preprocessing, model training, and visualization of feature contributions.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

Run the script with:

```bash
python feature_importance_LightGBM.py
```

## Features

- Data preprocessing with multiple scalers (StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer)
- Model training using LightGBM (`LGBMClassifier`)
- Feature importance evaluation (including permutation importance)
- Visualization with Matplotlib
- Support for cross-validation (StratifiedGroupKFold)
- Integration with PyTorch for advanced processing

## Files

- `feature_importance_LightGBM.py`: Main script
- `requirements.txt`: List of dependencies
- `README.md`: Project description

## License

This project is provided as-is under the MIT License (adjust if needed).
