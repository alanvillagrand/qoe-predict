# QoE-Predict: Quality of Experience Prediction for Point Cloud Streaming
This repository contains a machine learning pipeline for predicting Quality of Experience (QoE) in 3D point cloud streaming using 2D projection-based analysis.

The trained QoE model used in the projection experiments is provided here:
```
models/qoe_model_xgb_v3.0.pkl
```

The rest of this README contains instructions to run the pipeline to train the model and the projection-based evaluation method.

---

## 1. Environment Setup

This project requires Python 3.10+ and Conda.

### Create the environment

```bash
conda create -n qoe-predict python=3.10
conda activate qoe-predict
```

### Install dependencies

```bash
pip install -r requirements.txt
```

**NOTE:** If you want to run the projection method, you will need to install ffmpeg with VMAF support. A comprehensive guide on installing FFmpeg with VMAF can be found here: https://ottverse.com/vmaf-ffmpeg-ubuntu-compilation-installation-usage-guide/

---

## 2. Repository Structure

```
qoe-predict/
│
├── extract_data_to_csv.ipynb      # Extract LIVE-NFLX-II .mat files into training CSV
├── train_model.ipynb              # Train XGBoost QoE prediction model
├── cloud_projection.py            # Run projection-based QoE prediction on point clouds
├── evaluation_results.ipynb       # Evaluate prediction and projection experiment results
│
├── matdata/                       # Subjective video data from LIVE-NFLX-II dataset
├── data/                          # Output CSVs from extraction and projection
├── models/                        # Trained models (v3.0 used in paper results)
└── requirements.txt               # Python package requirements
```

---

## 3. Step-by-Step Usage

### Step 1 — Extract LIVE-NFLX-II Data

Use the notebook:

```
extract_data_to_csv.ipynb
```

This notebook:

- Loads all `.mat` files from `matdata/`
- Extracts frame-level VMAF, PSNR, SSIM, bitrate, rebuffering, MOS, etc.
- Produces a **single training CSV** stored in `data/`

Run each cell in order.

---

### Step 2 — Train the QoE Model

Use the notebook:

```
train_model.ipynb
```

This notebook:

- Loads the extracted CSV  
- Trains an **XGBoost regressor** to predict continuous MOS  
- Evaluates model performance (RMSE, R², Pearson, Spearman)  
- Saves the trained model into `models/`

The final model used in this project is:

```
models/qoe_model_xgb_v3.0.pkl
```

---

### Step 3 — Run the 3D Projection-Based QoE Experiment

The projection experiment requires point cloud video sequences from the Microsoft Voxelized Upper Bodies (MVUB) dataset.

Create the point cloud directory:

```bash
mkdir -p data/point_cloud
cd data/point_cloud
```

Download each point cloud sequence:

```bash
wget https://plenodb.jpeg.org/pc/microsoft/andrew9.zip && unzip andrew9.zip && rm andrew9.zip
wget https://plenodb.jpeg.org/pc/microsoft/david9.zip && unzip david9.zip && rm david9.zip
wget https://plenodb.jpeg.org/pc/microsoft/phil9.zip && unzip phil9.zip && rm phil9.zip
wget https://plenodb.jpeg.org/pc/microsoft/ricardo9.zip && unzip ricardo9.zip && rm ricardo9.zip
wget https://plenodb.jpeg.org/pc/microsoft/sarah9.zip && unzip sarah9.zip && rm sarah9.zip
```

Run the script:

```
cloud_projection.py
```

This script:

- Loads point cloud frames from directories such as `data/point_cloud/<subject>/`
- Renders six 2D projections per frame (front/back/left/right/top/bottom)
- Computes projection-based 2D quality metrics (PSNR, SSIM, VMAF optionally)
- Inputs metrics into the trained QoE model
- Outputs a CSV with predicted QoE per frame per sampling density

Run via:

```bash
python cloud_projection.py
```

Output is stored in:

```
data/pc_projection_qoe.csv
```

---

### Step 4 — Evaluate Projection Experiment Results

Use the notebook:

```
evaluation_results.ipynb
```

This notebook:

- Loads the projection-based QoE CSV  
- Computes descriptive statistics  
- Plots QoE vs. sampling density for each subject  
- Generates the figures used in the research paper  

Run all cells to reproduce results.

---
