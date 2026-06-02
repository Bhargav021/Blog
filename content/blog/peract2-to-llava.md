---
title: From PerAct2 to LLaVa for Bimanual VLA
tags:
  - blog
  - draft
  - robotics
  - vla
  - diffusion-models
  - robotics-research
status: draft
---

Planned post on dataset conversion, dual-gripper heatmaps, and iterable dataset fixes.

# Training a ControlNet Model for Bimanual Robotic Manipulation Heatmap Prediction

## Overview

This project explores the use of diffusion models, specifically ControlNet, for predicting structured action representations in a bimanual robotic manipulation setting.

The objective is to learn a mapping from:

- Multi-view visual observations (four camera views)
    
- Task instruction (text prompt)
    

to:

- Spatial heatmaps encoding gripper positions and orientations
    

This problem is relevant in robotics, where translating perception into structured intermediate representations is a key step toward actionable control.

The implementation includes:

- Synthetic dataset generation
    
- Attempted integration with a real dataset (Hugging Face)
    
- A ControlNet-based training and inference pipeline
    
- Dataset validation and debugging tools
    

---

## Problem Statement

The task is to predict bimanual manipulation targets in image space.

### Inputs

- A 2×2 tiled image combining four camera views (front, top, left, right)
    
- A natural language instruction (e.g., "Lift the ball")
    

### Outputs

- A heatmap image encoding:
    
    - Left gripper position (Gaussian blob)
        
    - Right gripper position
        
    - Gripper orientations (colored directional lines)
        

### Constraints

- Limited dataset size (synthetic dataset ~100 samples)
    
- Inconsistent quality in real dataset
    
- No explicit supervision separating spatial components
    
- Use of pretrained diffusion backbone (Stable Diffusion + ControlNet)
    

---

## Dataset

### Synthetic Dataset

A custom dataset is generated programmatically:

- Scene includes workspace, object, and two grippers
    
- Four camera views are rendered and tiled
    
- Heatmaps encode position (Gaussian) and orientation (lines)
    

Each sample contains:

- Conditioning image
    
- Target heatmap
    
- Task instruction
    
- Gripper pose metadata
    

### Real Dataset (Hugging Face)

An external dataset (`asmitamohanty/bimanual`) was explored:

- Multiple tasks available
    
- Data stored as ZIP archives
    

Observed issues:

- Corrupted image files
    
- Inconsistent directory structure
    
- Missing or unclear target formats
    

Only partial subsets were usable after validation.

[VISUALIZATION_PLACEHOLDER: dataset_overview]  
Description: Example tiled conditioning images and corresponding heatmaps

---

## Methodology

### Data Preprocessing

Two preprocessing approaches were implemented.

#### Initial Pipeline

- Resize images to 512×512
    
- Convert to tensor
    
- Normalize to [-1, 1]
    

#### Improved Pipeline

- Separate transforms for:
    
    - Conditioning images
        
    - Heatmaps
        
    - Visualization
        
- Optional Canny edge detection for conditioning
    
- Dataset validation including:
    
    - Image integrity checks
        
    - Directory cleanup
        
    - Corruption detection
        

Not implemented or unclear:

- Data augmentation
    
- Spatial alignment validation
    
- Explicit heatmap channel decomposition
    

[VISUALIZATION_PLACEHOLDER: preprocessing_pipeline]  
Description: Flow from raw data to model input

---

### Model / Approach

The model is based on Stable Diffusion with ControlNet.

Components:

- VAE for latent encoding
    
- UNet for noise prediction
    
- ControlNet for conditioning on images
    
- CLIP text encoder for prompts
    

Key design decisions:

- Freeze base model components initially
    
- Train ControlNet
    
- Later experiments allow UNet fine-tuning
    
- Use FP32 initially, later introduce mixed precision
    

Training objective:

- Predict noise in latent space using MSE loss
    

[VISUALIZATION_PLACEHOLDER: model_architecture]  
Description: ControlNet diffusion pipeline with conditioning inputs

---

### Training Procedure

Training loop:

- Encode heatmaps to latent space
    
- Add noise via scheduler
    
- Condition on image and text
    
- Predict noise
    
- Compute MSE loss
    

Configuration (varies across runs):

- Learning rate: ~1e-5
    
- Batch size: 1–2
    
- Steps: up to ~5000
    
- Gradient clipping enabled
    
- Cosine scheduler
    

Validation:

- Limited batches
    
- Early implementation contained incorrect validation loss
    
- Later version partially corrected
    

Checkpointing:

- Periodic saves
    
- Best model selected using training loss (not validation)
    

---

## Experiments & Iterations

Several experiments were conducted:

### Synthetic Data Training

- Verified end-to-end pipeline functionality
    

### NaN Loss Fix

- Switched to FP32
    
- Added checks for invalid values
    

### Dataset Debugging

- Multiple extraction strategies
    
- File validation and corruption analysis
    

### Canny Edge Conditioning

- Tested edge maps as conditioning input
    

### UNet Fine-Tuning

- Allowed training of UNet alongside ControlNet
    

### Improved Training Pipeline

- Added mixed precision
    
- Resume capability
    
- Visualization during validation
    

[VISUALIZATION_PLACEHOLDER: experiment_comparisons]  
Description: Qualitative or loss comparisons across configurations

---

## Results

Implemented metrics:

- Mean Squared Error (MSE)
    
- Mean Absolute Error (MAE)
    
- PSNR
    
- SSIM
    

However:

- No consolidated results table is available
    
- No systematic comparison across experiments
    
- Metrics are computed but not consistently analyzed
    

Observations:

- Visual outputs show partial alignment between prediction and ground truth
    
- Reliability is unclear due to dataset and validation issues
    

[VISUALIZATION_PLACEHOLDER: results_metrics]  
Description: Metric summaries or prediction comparisons

---

## Key Insights

- ControlNet can learn structured outputs from image inputs
    
- Dataset quality strongly impacts training stability
    
- Text conditioning is ineffective due to lack of prompt diversity
    
- Synthetic data is useful for debugging but limited for generalization
    

---

## Limitations

- Training and validation datasets are not properly separated
    
- Real dataset contains corrupted samples
    
- Validation procedure was incorrect in early implementation
    
- No data augmentation
    
- Weak supervision for structured outputs
    
- Limited reproducibility controls
    
- Multiple redundant pipeline implementations
    

---

## Future Work

- Standardize dataset and remove corrupted samples
    
- Implement proper train/validation/test splits
    
- Add augmentation strategies
    
- Improve supervision (separate channels, structured losses)
    
- Perform controlled experiments
    
- Improve evaluation and logging
    
- Extend pipeline to action generation
    

---

## Conclusion

This project establishes a working pipeline for using ControlNet in a bimanual manipulation setting, focusing on predicting structured heatmaps from visual input.

While core components are implemented, the system remains a prototype. Issues in dataset quality, validation design, and experimental rigor limit its reliability.

The work provides a foundation for further development toward a more robust and research-grade system.

---

## Appendix

### Example Training Step

```python
latents = vae.encode(target_images).latent_dist.sample()
noise = torch.randn_like(latents)
timesteps = torch.randint(0, T, (batch_size,))
noisy_latents = scheduler.add_noise(latents, noise, timesteps)

model_pred = unet(
    noisy_latents,
    timesteps,
    encoder_hidden_states=text_embeddings,
    controlnet_cond=conditioning_images
)

loss = mse(model_pred, noise)
```
