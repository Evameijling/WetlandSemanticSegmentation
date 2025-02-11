# AI-Driven Segmentation and Classification of Vegetation in the Biesbosch Floodplain Using Remote Sensing Data

This repository contains the code and methodology from my thesis research on **semantic segmentation for wetland vegetation classification**. The study explores **self-supervised learning (SSL) and deep learning architectures, such as U-Net**, to classify vegetation roughness using **Sentinel-2 and Very High-Resolution (VHR) Pléiades NEO imagery**. The goal is to enhance classification accuracy while **reducing reliance on annotated data**.

## 📌 Overview
The research focuses on:
- 🌍 **Remote sensing for wetland monitoring**, particularly in the Biesbosch area.
- 🧠 **Deep learning-based land cover classification** using U-Net, trained with and without SSL pretraining.
- 🔍 **Impact of image resolution on classification performance**.
- ⚙️ **Automated data preprocessing and segmentation pipeline** for high-resolution satellite imagery.

## 📂 Code Structure
This repository is a **restructured version** of my original research code, reorganized for clarity. Some file paths and references might differ from the original; however, with minor adjustments, all scripts remain fully functional on any system.

## 🛠 Installation & Usage

### General Dependencies
For all experiments, install the primary dependencies:
```bash
pip install -r requirements.txt
```
For scripts that retrieve satellite data from Google Earth Engine (GEE), install the additional dependencies:
```bash
pip install -r requirements_GEE.txt
```
**Note:** Ensure you have your GEE credentials set up since the scripts require personal API authentication.

## 📡 Data Sources

### Sentinel-2 Data and Labels:
- Copernicus Open Access Hub or
- Google Earth Engine Sentinel-2 Collection

### VHR Data:
- Download from [Satellietdataportaal](https://viewer.satellietdataportaal.nl/@52.06262,4.691162,8)

## 🚀 Running Experiments

### 1. Baseline U-Net Experiment (Medium Resolution)
#### Download Data:
Download Sentinel-2 data and corresponding labels from the Sentinel-2 data portal.
#### Preprocess Data:
Run the preprocessing scripts in the following order (located in `srs/dataset/medium_resolution/wetlands`):
*(Scripts are expected to be executed in the given order as per the repository instructions.)*
#### Train the Baseline U-Net Model:
Execute the following command to run the baseline U-Net experiment:
```bash
python srs/training/medium_resolution/biesbosch/train_unet_nopretrain_16bit_S2.py
```
This script outputs per-epoch metrics for the training and validation sets and the final test set metrics. Results can also be viewed in TensorBoard.

### 2. Experiment 1: Effect of Pretraining
#### On Medium-Resolution Data
##### Download Data:
Download Sentinel-2 data and labels from the Sentinel-2 data portal.
##### Preprocess Data:
Run the preprocessing scripts in order (located in `srs/dataset/medium_resolution/wetlands`).
##### Pretrain the Autoencoder:
Save weights for U-Net using:
```bash
python srs/training/medium_resolution/wetlands/train_autoencoder_save_weights_for_unet_loss_16bit_mixedloss.py
```
##### Train U-Net Models:
**Non-pretrained U-Net:**
```bash
python srs/training/medium_resolution/wetlands/train_unet_nopretrain_16bit.py
```
**Pretrained U-Net:**
```bash
python srs/training/medium_resolution/wetlands/train_unet_16bit.py
```
Both scripts output training and validation metrics per epoch and final test metrics at the last epoch. Metrics can be monitored with TensorBoard.

#### On Very-High Resolution Data
##### Download Data:
Download VHR data and labels from satellietdataportaal.
##### Preprocess Data:
Run the preprocessing scripts in order (located in `srs/dataset/high_resolution`).
##### Pretrain the Autoencoder:
Execute:
```bash
python srs/training/high_resolution/train_autoencoder_save_weights_for_unet_loss_16bit_mixedloss.py
```
##### Train U-Net Models:
**Non-pretrained U-Net:**
```bash
python srs/training/high_resolution/train_unet_nopretrain_16bit_SDP_1024.py
```
**Pretrained U-Net:**
```bash
python srs/training/high_resolution/train_unet_pretrain_16bit_SDP_1024.py
```
Similar to the medium-resolution case, these scripts log metrics per epoch and display the final test metrics. Use TensorBoard for visualization.

### 3. Experiment 2: Dependency on Resolution
#### Medium-Resolution Data (Wetlands Focus)
##### Download Data:
Download Sentinel-2 data and labels from the Sentinel-2 data portal.
##### Preprocess Data:
Run the preprocessing scripts located in `srs/dataset/medium_resolution/biesbosch`. This process focuses solely on wetlands and involves downsampling the corresponding high-resolution labels.
##### Train U-Net Models:
**Non-pretrained U-Net:**
```bash
python srs/training/medium_resolution/biesbosch/train_unet_nopretrain_16bit_S2.py
```
**Pretrained U-Net:**
```bash
python srs/training/medium_resolution/biesbosch/train_unet_pretrain_16bit_S2.py
```
#### High-Resolution Data
Follow the same procedure as described in **Experiment 1: On Very-High Resolution Data**.

## 🔧 Additional Utilities

### Utilities:
Scripts in `srs/utils` help manage files and evaluate satellite and mask files.
### Models:
The `srs/models` directory contains model definitions used in the experiments.
### Evaluation:
Use the script in `srs/evaluation` to run inference on U-Net given the model weights.

## 📜 Citation

If you use this repository in your research, please cite it using the following BibTeX entry:
```bibtex
@misc{meijling2025wetlandssegmentation,
  author    = {Gmelich Meijling, Eva},
  title     = {AI-Driven Segmentation and Classification of Vegetation in the Biesbosch Floodplain Using Remote Sensing Data},
  journal   = {Journal of Wetland Remote Sensing},
  year      = {2025},
  url       = {https://github.com/Evameijling/WetlandSemanticSegmentation}
}
```

## 📧 Contact
For questions or collaborations, please open an issue on GitHub or contact me directly.
EOF
