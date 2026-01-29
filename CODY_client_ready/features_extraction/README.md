# Feature extraction with YOLOv8-Pose

This folder contains the **feature extraction pipeline** used to convert raw videos into structured datasets of pose features using **YOLOv8-Pose**.

The main entry point is:

- `video_features_extraction.ipynb`

## Folder structure

```
features_extraction/
├── video_features_extraction.ipynb   # main notebook
├── videos_dir/                       # input folder for raw .mp4 videos
└── outputs/                          # outputs (annotated videos + exported Excel time series)
```

## Recommended environment

The notebook can be executed locally or on **Google Colab** (GPU recommended).

Typical dependencies include `ultralytics`, `torch`, `opencv-python`, and `pandas`.  
For local installs, you can use the repository’s `requirements-extraction.txt`:

```bash
pip install -r requirements-extraction.txt
```

## YOLO weights

If you use a YOLOv8 pose model that requires local weights (e.g., `yolov8x-pose-p6.pt`), place the weights file in this folder and update the notebook cell that loads the model accordingly.

## Usage

1. Put input videos into `videos_dir/`.
2. Run `video_features_extraction.ipynb`.
3. For each input video, the notebook typically produces:
   - an **annotated video** (keypoints overlay) in `outputs/`
   - an **Excel file** (`.xlsx`) containing pose time series (and/or derived signals) in `outputs/`
