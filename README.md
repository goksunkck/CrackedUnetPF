# Fatigue Crack Tip Detection and Tracking

This project implements a hybrid framework for precise fatigue crack tip detection and growth tracking. It combines a deep learning-based segmentation model (**Attention U-Net**) with a physics-informed **Particle Filter** to robustly estimate crack length and material parameters (Paris Law constants) over time.

## Overview

Accurate tracking of fatigue cracks is critical for structural health monitoring. This approach leverages:
1.  **Deep Learning (CNN)**: An Attention U-Net model segments the crack tip from displacement/strain field data (Nodemaps), providing noisy but direct measurements of the crack position.
2.  **Particle Filter (PF)**: A probabilistic filter that assimilates the CNN measurements with a physical crack growth model (Paris Law). This allows for:
    -   Smoothing of noisy CNN detections.
    -   Estimation of unobservable material parameters ($C$ and $m$).
    -   Prediction of future crack growth.

## Project Structure

```
├── data/               # Raw and processed datasets
├── outputs/            # Generated plots and tracking results
├── scripts/            # Execution scripts (e.g., main_cnn_pf.py, train_attention.py)
├── src/                # Core source code
│   ├── particle_filter.py  # Particle Filter implementation
│   ├── model_attention.py  # Attention U-Net architecture
│   └── dataset.py          # Data loaders
├── make_data.py        # Data preparation script
└── README.md
```

## Installation

Ensure you have Python 3.x installed along with the following dependencies:

```bash
pip install torch numpy matplotlib
```

## Dataset

The dataset used in this project can be downloaded from Zenodo:
[https://zenodo.org/records/5740216](https://zenodo.org/records/5740216)

**Setup Instructions:**
1. Create a `data` directory in the project root if it doesn't exist.
2. Download the dataset folders (e.g., `S_160_4.7`, `S_160_2.0`, `S_950_1.6`) from the link.
3. Extract/place them inside the `data` directory such that the structure looks like:
   ```
   data/
   ├── S_160_4.7/
   ├── S_160_2.0/
   └── S_950_1.6/
   ```

## Usage

### 1. Data Preparation
Before running the tracker, raw data (Nodemaps) must be processed into PyTorch tensors.

```bash
python make_data.py --experiment S_160_4.7
```
*Available experiments: `S_160_4.7`, `S_160_2.0`, `S_950_1.6`*

### 2. Running the Tracker
To run the full crack tracking pipeline (CNN detection + Particle Filter estimation):

```bash
python scripts/main_cnn_pf.py
```
This script will:
- Load the pre-trained Attention U-Net (`checkpoints/attention_unet_best.pth`).
- Iterate through the processed datasets.
- Perform frame-by-frame tracking.
- Save result plots to the `outputs/` directory.

## Methodology

### Deep Learning Model
An **Attention U-Net** is employed to focus on relevant spatial features representing the crack tip. The model takes 2-channel inputs (e.g., global displacement fields) and outputs a segmentation mask identifying the crack tip location.

### Particle Filter
A Sequential Importance Resampling (SIR) Particle Filter is used for state estimation.
- **State Vector**: $[a, \log(C), m]$
  - $a$: Crack length
  - $C, m$: Paris Law constants
- **Transition Model**: Based on Paris Law:
  $$ \frac{da}{dN} = C (\Delta K)^m $$
- **Observation Model**: Likelihood is computed based on the Euclidean distance between the Particle predictions and the CNN measured crack tip.

## Reference

This work is based on the datasets described in:

> **D. Melching, T. Strohmann, G. Requena, and E. Breitbarth, “Explainable machine learning for precise fatigue crack tip detection,” Scientific Reports, vol. 12, no. 1, p. 9513, 2022.**
