# Temporal Phase-Difference Guided Spatiotemporal Learning for DSA Vessel Segmentation

This repository contains the official implementation of our paper:

**“Temporal Phase-Difference Guided Spatiotemporal Learning for DSA Vessel Segmentation”**

## 🔍 Overview

Vessel segmentation in Digital Subtraction Angiography (DSA) plays a critical role in quantitative vascular analysis and interventional planning. However, accurate segmentation remains challenging due to:
- complex and fine-grained vessel structures (e.g., thin distal branches),
- strong background interference from static tissues and bones,
- and the dynamic nature of contrast agent flow across time.

To address these challenges, we propose a **spatiotemporal DSA vessel segmentation framework** that explicitly integrates **hemodynamic priors** with **efficient long-range temporal modeling**.

## ✨ Key Contributions

- **Framework**
 ![Network Framework] (figures/framework.pdf)

- **Mamba-based Temporal Encoder**  
  We adopt a Mamba-based state space model to capture blood flow dynamics and long-range temporal dependencies in DSA sequences with **linear computational complexity**, enabling efficient and scalable temporal modeling.

- **Forward Phase-Difference Maximum Intensity Projection (FPD-MaxIP)**  
  Inspired by contrast agent flow characteristics, we construct an explicit phase-difference prior from early and late DSA frames, which:
  - enhances contrast-responsive vessel structures,
  - suppresses temporally persistent background tissues.

- **Minimum Intensity Projection (MinIP) as a Spatial Prior**  
  A complementary MinIP representation is introduced to encode coarse static vessel morphology, providing robust spatial guidance.

- **Domain-aware Fusion Module**  
  We design a fusion module that explicitly models interactions between temporal features and spatial priors before decoding, improving vessel connectivity and fine-structure preservation.

## 📊 Experimental Results

Extensive experiments on the **[DIAS](https://zenodo.org/records/11637181)** and **[DSCA](https://zenodo.org/records/11255024)** datasets demonstrate that our method consistently outperforms state-of-the-art approaches, particularly in:

![Results on DIAS dataset and DSCA dataset](figures/vis.pdf)
- preserving vessel connectivity,
- accurately segmenting thin and distal vessel branches,
- and reducing background-induced false positives.

## 📁 Instructions

More detailed instructions for data preparation, training, and evaluation are provided below.

### Train

### Test

# Acknowledgement

