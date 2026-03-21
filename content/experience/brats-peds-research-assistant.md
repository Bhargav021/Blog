---
title: Research Assistant - BraTS-PEDs
tags:
  - experience
  - medical-ai
  - segmentation
---

- Organization: Pediatric Brain Tumor Segmentation (BraTS-PEDs 2025)
- Location: Los Angeles, CA
- Period: Jul 2025 - Present

## What problem we solved

Pediatric brain tumors require precise MRI segmentation for treatment planning, but class imbalance and noisy annotations make this difficult.

## What I built

- Engineered end-to-end preprocessing for 1000+ MRI volumes with TorchIO, SimpleITK, Nyul normalization, and N4 bias correction
- Achieved greater than 95% consistency across scans
- Architected GPU-efficient 3D models (DeepDenseTrans3D, GIETNet, ARES-UNet) with attention mechanisms

## Measurable impact

- Boosted Dice score by 8% over baseline U-Net
- Composite loss (AFTL + Boundary + IoU + BCE) reduced false negatives by 15%
- AMP and DDP optimization reduced training runtime by 25%

## What I learned

Preprocessing and loss design can matter as much as architecture in medical imaging.
