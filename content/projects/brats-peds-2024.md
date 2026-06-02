---
title: BraTS-PEDs 2024 Pediatric Brain Tumor Segmentation
tags:
  - projects
  - medical-ai
  - segmentation
  - medical-imaging
  - 3d-segmentation
  - pytorch
---

- Tagline: Deep learning for clinical MRI analysis
- Category: Featured Research
- GitHub: [BraTs Challenge](https://github.com/Bhargav021/BraTs-Challenge-6)

## Context

High-stakes multimodal MRI segmentation with class imbalance and noisy labels.

## Role

Lead Researcher responsible for preprocessing, architecture, training strategy, and evaluation.

## Work completed

- Preprocessed T1/T2/FLAIR using TorchIO normalization, resampling, and augmentation
- Implemented 3D U-Net with attention gates and deep supervision
- Optimized with Dice + Focal loss
- Trained on A100 with AMP and validation-based early stopping

## Results

- Dice > 0.85 for whole tumor
- Dice > 0.78 for tumor core
- 15% reduction in false negatives
- ~8% improvement over baseline
