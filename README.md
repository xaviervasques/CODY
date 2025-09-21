# CODY

This repository contains scripts for **feature extraction, classification, and analysis** of patient video data using machine learning.

## 📂 Project Structure
- `1_features_extraction/` → Extract features from videos (YOLO-based, notebooks).
- `2_combined_classification/` → Combined classification windows.
- `3_patients/` → Patient-level classification.
- `4_windows/` → Symptom vs control classification using windows.
- `5_tasks/` → Symptom vs control classification using tasks.
- `6_feature_importance/` → Feature importance analysis.
- `datasets/` → Datasets.

## 🚀 Installation
Clone the repository:
```bash
git clone https://github.com/your-username/Code_GEMP.git
cd Code_GEMP
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Usage
Each subfolder has its own scripts and examples. For example:
```bash
python 2_combined_classification/combined_windows.py
```

## 🎯 Model Weights
This project requires **YOLOv8 pose weights** (`yolov8x-pose-p6.pt`) for feature extraction.  
Due to GitHub’s file size limit (>100 MB), the weights are **not stored in the repository**.

Please download them manually from the [Ultralytics YOLOv8 releases](https://github.com/ultralytics/ultralytics) or other sources and place the file at:
```
1_features_extraction/yolov8x-pose-p6.pt
```

## 📜 License
MIT License (or update as appropriate).
