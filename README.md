# pose_estimation

This project develops an automated computer vision pipeline for analyzing chicken gait patterns to detect lameness. The system captures videos of chickens walking through a corridor using fixed-position cameras, extracts keypoint coordinates through a fine-tuned YOLOv8 pose estimation model trained on HPC clusters, and generates gait scores to assess lameness severity.

## System Overview

The pipeline processes videos of chickens walking in a controlled corridor environment:

- Input: Video recordings from fixed-position cameras showing chickens walking from one end of a corridor to the other
  
- Manual Annotation: Using CVAT manually annotates 200 representative images, including 8 keypoints(head, neck, hocks, feet, knees) and bounding boxes
  
- Keypoint Detection: A fine-tuned YOLOv8 model is used to predict the coordinates of the eight keypoints in each frame
  
- Feature Extraction: Gait features are computed over a time series based on the coordinate values
  
- Classification: The features are combined with expert-validated gait scores to form an ANOVA model
  
- Output: Numerical gait score and binary lameness classification

<img width="1010" height="240" alt="image" src="https://github.com/user-attachments/assets/b425044b-1da0-4276-9b59-c2b121a52341" />
Figure: The camera is fixed in the back-view position, recording a video of chickens walking through the corridor

<img width="1030" height="584" alt="image" src="https://github.com/user-attachments/assets/d0cdd8fe-d76a-4310-a1f4-81762c459220" />
Figure: System architecture pipeline showing the complete workflow from image input to final identification

## Key Features
- Automated Gait Analysis:
    Real-time keypoint detection using fine-tuned YOLOv8 pose estimation
    Comprehensive gait feature extraction from temporal sequences
    Expert-validated scoring algorithm for lameness classification

- HPC-Optimized Training:
    Model training on high-performance computing clusters
    SLURM job management for scalable training workflows

- Robust Data Processing:
    Automated video frame extraction with configurable sampling rates
    Statistical outlier detection and temporal interpolation
    CSV-based coordinate data management system

- Professional Annotation Workflow:
    CVAT integration for precise keypoint labeling
    8-keypoint anatomical landmark annotation protocol
    Quality control mechanisms for annotation consistency

- Statistical Analysis:
    ANOVA-based feature selection and model fitting
    Correlation analysis with expert gait assessments
    Performance metrics and validation protocols

- Production-Ready Pipeline:
    End-to-end automation from video input to score output
    Batch processing capabilities for multiple videos
    Configurable confidence thresholds and processing parameters


## Dataset Structure
The project follows a structured data organization optimized for computer vision workflows:

```bash

pose_estimation/
├── README.md
├── requirements.txt
├── environment.yml
│
├── data/                           # Data directory
│   ├── raw_videos/                 # Original video files
│   │   ├── chicken_001.mp4
│   │   ├── chicken_002.mp4
│   │   └── ...
│   │
│   ├── frames/                     # Extracted video frames
│   │   ├── train/                  # Training images
│   │   ├── val/                    # Validation images
│   │   └── test/                   # Testing images
│   │
│   ├── annotations/                # CVAT annotation files
│   │   ├── yolo_format/           # YOLO format labels
│   │   │   ├── train/
│   │   │   └── val/
│   │   ├── cvat_export.xml        # Raw CVAT export
│   │   └── data.yaml              # YOLOv8 dataset config
│   │
│   └── processed/                  # Processed datasets
│       ├── keypoints.csv          # Extracted coordinates
│       ├── features.csv           # Computed gait features
│       └── scores.csv             # Expert assessments
│
├── models/                         # Model files
│   ├── pretrained/
│   │   └── yolov8n-pose.pt        # Base YOLOv8 pose model
│   ├── checkpoints/               # Training checkpoints
│   └── production/
│       ├── best.pt                # Fine-tuned YOLOv8 model
│       └── gait_scorer.pkl        # Trained scoring model
│
├── src/                           # Source code
│   ├── data/
│   │   ├── extract_frames.py      # Video frame extraction
│   │   ├── preprocess.py          # Data preprocessing
│   │   └── augmentation.py        # Data augmentation
│   │
│   ├── models/
│   │   ├── train.py               # Model training script
│   │   ├── evaluate.py            # Model evaluation
│   │   └── predict.py             # Batch prediction
│   │
│   ├── features/
│   │   ├── extraction.py          # Feature calculation
│   │   ├── selection.py           # Feature selection
│   │   └── analysis.py            # Statistical analysis
│   │
│   └── utils/
│       ├── visualization.py       # Plotting utilities
│       ├── metrics.py             # Evaluation metrics
│       └── io.py                  # File I/O operations
│
├── scripts/                       # Execution scripts
│   ├── run_model_hpc.sh          # SLURM training job
│   ├── batch_predict.sh          # Batch processing
│   └── pipeline.sh               # Full pipeline execution
│
├── configs/                       # Configuration files
│   ├── training.yaml             # Training hyperparameters
│   ├── model.yaml                # Model architecture
│   └── pipeline.yaml             # Pipeline settings
│
├── results/                       # Output directory
│   ├── predictions/              # Model predictions
│   ├── visualizations/           # Generated plots
│   ├── reports/                  # Analysis reports
│   └── logs/                     # Training/execution logs
│
└── docs/                         # Documentation
    ├── annotation_guide.md       # Keypoint annotation protocol
    ├── training_guide.md         # HPC training instructions
    └── api_reference.md          # Code documentation

```

## Data Format Requirements
- Video Input Format:
  - Resolution: 1080p (1920×1080)
  - Frame Rate: 60 FPS
  - Duration: 10-60 seconds per recording

- Chicken ID and Expert Gait Assessment Format:
```bash

chicken_id,video_name,start_timestamp,end_timestamp,gait_score
CHK001,chicken_001.mp4,0.000,12.450,0
CHK002,chicken_002.mp4,1.200,15.670,1

```
 
- Keypoint Annotation Standard:
```bash

# Keypoint Definition (0-indexed)
KEYPOINT_MAPPING = {
    0: "head",          # Head center point
    1: "neck",          # Neck-body junction
    2: "left_hock",     # Left tarsal joint (ankle equivalent)
    3: "right_hock",    # Right tarsal joint
    4: "left_foot",     # Left foot pad center
    5: "right_foot",    # Right foot pad center  
    6: "left_knee",     # Left knee joint (tibiotarsal)
    7: "right_knee"     # Right knee joint
}

# Visibility Codes
VISIBILITY = {
    0: "not_visible",     # Keypoint outside image or occluded
    1: "visible",         # Keypoint visible may be ambiguous or unambiguous
}

```
  
- YOLO Format Annotation
Each annotation file contains normalized coordinates in YOLO pose format:
```bash

# Format: class_id center_x center_y width height x1 y1 v1 x2 y2 v2 ... x8 y8 v8
# Example annotation line:
0 0.687 0.699 0.277 0.434 0.851 0.441 1 0.853 0.457 1 0.755 0.623 1 0.963 0.598 1 0.766 0.709 1 0.904 0.713 1 0.724 0.814 1 0.890 0.780 1

# Where:
# - class_id: 0 (chicken)
# - center_x, center_y: bounding box center (normalized 0-1)
# - width, height: bounding box dimensions (normalized 0-1)
# - x1 y1 v1: head coordinates and visibility
# - x2 y2 v2: neck coordinates and visibility
# - ... (8 keypoints total)

```
