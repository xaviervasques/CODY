# 🎥 Feature Extraction with YOLOv8-Pose

This folder contains the **feature extraction pipeline** used in our study on hyperkinetic movement disorders.  
It converts raw clinical videos into structured datasets of human pose features using **YOLOv8-Pose**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YourUserName/YourRepoName/blob/main/1_feature_extraction/member_tremor_all_videos.ipynb)

---

## 📂 Folder Structure

```
1_feature_extraction/
│
├── video_features_extraction.ipynb  # Main notebook for feature extraction
├── yolov8x-pose-p6.pt               # Pretrained YOLOv8 Pose model (17 keypoints)
├── video_dir/                       # Input folder for raw `.mp4` videos
└── outputs/                         # Output folder (annotated videos & features in Excel)
```

---

## ⚙️ Requirements & Environment

All required libraries (`ultralytics`, `torch`, `opencv-python`, `pandas`, etc.)  
are installed directly inside the notebook.  

- ✅ **Recommended**: Run on **Google Colab with GPU (CUDA)** for best performance.  

---

## 🎯 Model Weights
This project requires **YOLOv8 pose weights** (`yolov8x-pose-p6.pt`) for feature extraction.  
Due to GitHub’s file size limit (>100 MB), the weights are **not stored in the repository**.

Please download them manually from the [Ultralytics YOLOv8 releases](https://github.com/ultralytics/ultralytics) or other sources and place the file at:
```
1_features_extraction/yolov8x-pose-p6.pt
```

## ▶️ Usage

1. Upload your input videos to the `video_dir/` folder.  
2. Ensure that the YOLOv8 Pose model file `yolov8x-pose-p6.pt` is available in this directory.  
3. Open and run the notebook `video_features_extraction.ipynb` in Google Colab.  

For each input video:  
- An **annotated video** with detected keypoints is saved in `outputs/`.  
- A **time-series dataset** (`.xlsx`) with keypoint coordinates and derived distances is saved in `outputs/`.  

---

## 📊 Example Output

For an input video `20240703_105018.mp4`, the following files will be generated:  

- Annotated video → `outputs/20240703_105018_yolo.mp4`  
- Extracted features → `outputs/20240703_105018_timeseries.xlsx`  

---

## 📜 Citation

If you use this code or dataset in your research, please cite our related work:  

> *Pose-Based Deep Learning for Simultaneous Symptom Recognition in Hyperkinetic Movement Disorders*  
