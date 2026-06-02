---
title: Inside BraTS-PEDs What Improved Segmentation
tags:
  - blog
  - draft
  - medical-ai
  - segmentation
  - pytorch
  - research-log
status: draft
---

Planned post covering preprocessing, loss balancing, and model choices that moved Dice performance.

# DenseTrans-PED: Iterative Development of a 2.5D-to-3D Brain Tumor Segmentation Pipeline for Pediatric MRI

---

## Overview

Pediatric brain tumor segmentation is one of the most clinically consequential and technically demanding tasks in medical image analysis. Accurate delineation of tumor sub-regions from multi-modal MRI directly informs surgical planning, radiotherapy targeting, and treatment response monitoring. Yet automated segmentation systems built for adult gliomas perform poorly on pediatric cases, where tumor morphology, spatial distribution, and sub-region characteristics differ substantially.

This project develops a deep learning segmentation pipeline targeting the **BraTS 2025 Challenge, Task 6 (Pediatric Brain Tumors)**. The approach evolves through ten distinct experimental iterations, beginning with a lightweight 2.5D slice-based UNet++ and culminating in a full 3D volumetric architecture augmented with Swin Transformer attention, attention-gated skip connections, few-shot meta-learning, hard negative mining, and uncertainty-guided sampling.

The work is implemented as a Google Colab notebook in PyTorch and represents an active, in-progress research effort. The pipeline is not yet complete: no quantitative results have been recorded and no challenge submission has been produced. This post documents the design decisions, experimental progression, and current state honestly and precisely.

---

## Problem Statement

The task is **five-class voxel-wise semantic segmentation** of pediatric brain MRI volumes into:

- Class 0: Background
- Class 1 (C1): Non-enhancing tumor core / necrosis
- Class 2 (C2): Peritumoral edema (uncertain — see Limitations)
- Class 3 (C3): Enhancing tumor
- Class 4 (C4): Non-enhancing tumor / infiltration

Two composite regions are also evaluated:
- **Whole Tumor (WT)**: union of all foreground classes
- **Tumor Core (TC)**: union of C1, C3, C4

The primary challenge metrics are **mean Dice Similarity Coefficient (DSC)** and **95th-percentile Hausdorff Distance (HD95)** per sub-region, consistent with the official BraTS evaluation protocol.

**Core constraints and difficulties:**

- Severe class imbalance: foreground classes represent as little as 0.002% of total voxels (reflected in hardcoded class weights of approximately 49× for C3 and C4 relative to background).
- Small and spatially diffuse lesions, particularly C3 and C4, that are frequently absent from randomly sampled training patches.
- Limited training data: the BraTS-PED dataset is considerably smaller than its adult counterpart, making generalization harder and few-shot techniques relevant.
- Computational constraints: early experiments ran on Colab free-tier GPUs; later experiments target a dedicated L4 GPU (22.5 GB VRAM).

---

## Dataset

**Source:** BraTS 2025 Challenge, Task 6 — Pediatric Brain Tumors (BraTS-PED). Data is accessed via Google Drive as preprocessed PyTorch `.pt` tensor files.

**Format:** Each `.pt` file contains a Python dictionary with two keys:
- `'image'`: a float tensor of shape `(C, D, H, W)` where `C` is the number of MRI modalities, and `D, H, W` are the spatial dimensions.
- `'label'`: a long integer tensor of shape `(D, H, W)` containing per-voxel class labels in `{0, 1, 2, 3, 4}`.

**Modalities:** Five channels are used (`IN_CHANNELS=5`), described in the notebook as T1, T1c, T2, FLAIR, and a fifth channel labeled "Grad." The identity of the fifth channel is not documented (see Limitations).

**Class distribution (inferred from hardcoded weights):**

| Class | Weight | Interpretation |
|-------|--------|----------------|
| 0 (Background) | 1.43 | Dominant class |
| 1 (C1) | 48.62 | Rare foreground |
| 2 (C2) | 41.12 | Rare foreground |
| 3 (C3) | 49.63 | Rare foreground |
| 4 (C4) | 48.65 | Rare foreground |

The class weights were hardcoded into the notebook; the voxel-counting pass that produced them is not present.

**Dataset size:** Not explicitly stated in the notebook. The per-epoch subsampling configuration (`SCANS_PER_EPOCH=5–15`) implies a dataset large enough that full-epoch iteration is memory-impractical, but the exact number of training scans is undocumented.

[VISUALIZATION_PLACEHOLDER: dataset_overview]
Description: Bar chart of approximate class voxel frequency (background vs. C1–C4), spatial dimension distribution across scans, and example axial slices showing each of the five modalities alongside the ground-truth segmentation mask.

---

## Methodology

### Data Preprocessing

The preprocessing pipeline has two distinct phases, only the second of which is implemented in this notebook.

**Phase 1 — NIfTI to tensor conversion (absent):**
The pipeline that converts raw BraTS NIfTI volumes to `.pt` tensors — including co-registration, skull stripping, resampling, and initial normalization — is not present in the notebook. The `.pt` files are consumed as given.

**Phase 2 — In-dataset preprocessing (implemented, evolving across cells):**

*Cells 0–5 (minimal):*
- Raw float values from `.pt` files are fed to the model without normalization.
- No explicit intensity range validation.

*Cells 7–8 (percentile normalization):*
- Each 3-slice modality group (prev/curr/next) is independently clipped to the 1st–99th percentile and rescaled to `[0, 1]`.
- Note: this normalization is computed per sample rather than per volume, introducing inter-slice intensity inconsistency within the same scan.

*Cell 9 (nnU-Net-style normalization):*
- Per-modality z-score normalization using training set statistics, following the nnU-Net preprocessing convention.
- Applied consistently at the volume level before patch extraction.

**2.5D slice stacking (Cells 0–8):**
For each target axial slice at index `i`, the adjacent slices at indices `i-1` and `i+1` are concatenated along the channel dimension, yielding an input tensor of shape `(15, H, W)` (5 modalities × 3 slices). Boundary slices are handled by repeating the edge slice. This provides limited 3D context without full volumetric convolutions.

**3D patch extraction (Cell 9):**
Randomly sampled 3D patches of a fixed spatial size are extracted from full volumes. Sampling is class-aware and hierarchical: patches containing C4 voxels are preferred, followed by C3 and C1, with background patches included at a configurable probability. Edge slices (bottom and top 10% of volume depth) are excluded from sampling to avoid mostly-empty patches.

**Augmentation pipeline:**

*Cells 0–5:* Random 2D spatial crop with zero-padding fallback.

*Cells 7–8 (`AdvancedAugmentation`):*
- Random horizontal flip (p=0.5)
- Random rotation ±15° via affine grid sampling (p=0.3)
- Random gamma correction in range [0.8, 1.2] (p=0.4)
- Additive Gaussian noise with σ=0.05 (p=0.2)

*Cell 9 (`IntensityAugmentation`):*
- Random gamma correction [0.7, 1.5]
- Random MRI bias field simulation
- Additive Gaussian noise
- Random Gaussian blur

[VISUALIZATION_PLACEHOLDER: preprocessing_pipeline]
Description: Flowchart showing the 2.5D stacking procedure: a 3D volume is sliced axially, three adjacent slices are extracted and concatenated channel-wise to form a 15-channel 2D tensor. A separate diagram illustrates the 3D hierarchical patch sampling strategy used in Cell 9, with class-weighted sampling probabilities annotated.

---

### Model Architecture

The architecture evolves substantially across ten cells. Three principal designs are described below.

#### Design 1 — DenseTrans2D (Cells 0–5)

A 2D UNet++ with dense skip connections and optional Swin Transformer blocks in the decoder.

**Encoder:** 3–4 levels of `DepthwiseSeparableConv2d` blocks (depthwise + pointwise Conv2d, InstanceNorm2d, ReLU) followed by MaxPool2d downsampling.

**Decoder:** UNet++ nested skip connections. Each decoder node (`UNetPlusPlusNode2D`) receives an upsampled feature map plus all same-scale encoder outputs from prior decoder columns, concatenated along the channel dimension. Gradient checkpointing is applied at each node to reduce activation memory.

**Swin Transformer blocks (Cells 3–5):** Window-partitioned self-attention with window size 8×8, applied as a residual branch after the convolutional path in each decoder node. Uses `einops.rearrange` for window partitioning.

**Attention gates (Cell 5):** A gating mechanism on each skip connection — the upsampled decoder feature serves as the gating signal, producing a sigmoid attention map that selectively weights the encoder skip features before concatenation.

**Filter progression:** `[16, 32, 64, 128]` (Cells 0, 2–4) or `[32, 64, 128, 256]` (Cells 1, 5).

**Output:** Deep supervision with 3 prediction heads at different decoder depths (`final1`, `final2`, `final3`), each a 1×1 Conv2d projecting to `OUT_CHANNELS=5`.

#### Design 2 — EnhancedDenseTrans2D (Cells 7–8)

An upgraded 5-level version targeting an L4 GPU.

**Key differences from Design 1:**
- Filter sizes `[48, 96, 192, 384, 768]` (Cell 8) — significantly larger capacity.
- `EnhancedDepthwiseSeparableConv2d`: replaces ReLU with GELU, adds `Dropout2d(0.12)`, and uses custom `LayerNorm2d` instead of InstanceNorm2d.
- `MultiScaleAttentionGate`: parallel 3×3 and 5×5 attention branches concatenated before the sigmoid, replacing the single-scale gate of Cell 5.
- `EnhancedSwinTransformerBlock2D`: adds a full MLP branch (consistent with the original Swin design), learnable residual scaling parameters γ1 and γ2 initialized at 0.1, and a corrected window partition/merge implementation.
- `EnhancedUNetPlusPlusNode2D`: adds a residual projection (`Conv2d` or `Identity`) from the concatenated input to the output, enabling gradient flow across skip-heavy nodes.
- Weight initialization: Kaiming normal for all Conv2d and ConvTranspose2d layers.
- 4 deep-supervision heads.

#### Design 3 — ImprovedUNet3D (Cell 9)

A full 3D volumetric architecture with meta-learning auxiliary components.

**Backbone:** 5-level 3D UNet with `ImprovedResBlock` units (3D Conv, GroupNorm, dilation, residual addition).

**CoarseGating (BroadGate):** A coarse tumor presence prediction from the bottleneck that gates the decoder feature maps. Activates whenever any foreground class is predicted, priming the decoder for all tumor sub-regions including C1.

**Auxiliary heads:**
- `C4PresenceHead`: a binary classifier on the bottleneck features predicting whether C4 is present in the patch. Trained with weighted BCE to handle C4 scarcity.
- `MetaPrototypeHead`: projects features into an embedding space, computes k-shot class prototypes from support examples within the batch, and produces a cosine-similarity classifier output. Generates two auxiliary losses: a meta-classification loss and a supervised contrastive loss (`SupervisedContrastiveLoss`) applied to C3 and C4 embeddings.
- `MultiPrototypeBank`: an EMA-updated prototype memory bank with K slots per class, enabling cross-batch prototype refinement.

**Auxiliary losses (training only):**
- `SoftTopologicalLoss`: max-pooling-based topological regularization encouraging spatially coherent predictions.
- `EntropyFocusedLoss`: penalizes high-entropy (uncertain) predictions specifically on voxels belonging to rare classes (C1, C3, C4).

**Output:** Three prediction heads (`ds1`, `ds2`, `main_out`) with deep-supervision annealing — intermediate head weights decay to zero after a configured epoch threshold.

[VISUALIZATION_PLACEHOLDER: model_architecture]
Description: Three-panel diagram. Panel 1: DenseTrans2D UNet++ topology showing 4-level encoder, nested decoder nodes (x00 through x03), attention gates on skip connections, and 3 output heads. Panel 2: EnhancedDenseTrans2D block diagram highlighting the 5-level structure and enhanced component stack. Panel 3: ImprovedUNet3D schematic showing the 3D encoder, CoarseGating, three auxiliary heads (C4PresenceHead, MetaPrototypeHead, MultiPrototypeBank), and the three deep-supervision outputs.

---

### Training Procedure

**Hardware:** Google Colab (Cells 0–6, free tier, likely T4 GPU) and a dedicated L4 GPU with 22.5 GB VRAM and 58 GB RAM (Cells 7–9).

**Optimizer:**
- Cells 0–5: Adam, `lr=4e-4`, weight decay varies (0 or 1e-5).
- Cells 7–8: AdamW, `lr=8e-5`, `weight_decay=5e-7`, `betas=(0.9, 0.999)`.
- Cell 9: AdamW, `lr=8e-5` initial, `weight_decay` configured.

**Learning rate schedule:**
- Cells 1–5: `CosineAnnealingLR(T_max=EPOCHS)`.
- Cells 7–8: `OneCycleLR` with `max_lr=3e-4`, 15% warmup, cosine annealing. Note: `total_steps` is approximated by a fixed estimate of slices-per-scan and is not exact.
- Cell 9: `LambdaLR` with linear warmup (`WARMUP_EPOCHS`) followed by cosine decay. Additionally, a manual LR reduction by a fixed factor is applied when WT Dice plateaus for a configured number of epochs.

**Mixed precision:** `torch.amp.autocast` and `GradScaler` enabled for all CUDA runs from Cell 0. NaN loss and NaN gradient detection with batch skipping added from Cell 7.

**Gradient clipping:** `clip_grad_norm_` applied from Cell 7, with `max_norm=0.8` (Cells 7–8) and `max_norm=1.0` (Cell 9).

**Gradient accumulation:** `ACCUM_STEPS` batches accumulated before an optimizer step (Cell 9 only). The segmentation loss is divided by `ACCUM_STEPS`; auxiliary losses are not.

**Batch size:** 16 (Cells 0–5), 6 (Cells 7–9, reduced for L4 GPU memory constraints).

**Epochs:** 100 (Cell 0), 200 (Cells 4–5), 300 (Cells 7–9).

**Validation strategy:**
- Cell 0: No validation split — training set used for both.
- Cells 1–3: Single fixed scan held out as validation.
- Cells 4–9: 80/20 `train_test_split` (`random_state=42`, sklearn). Per-epoch random subsampling of `SCANS_PER_EPOCH` training scans and `VAL_SCANS_PER_EPOCH` validation scans.

**Early stopping:** From Cell 4 onward, training stops if the monitored validation metric (val loss in Cell 4; val Dice in Cells 7–8; combined score in Cell 9) does not improve for a patience period of 7 (Cell 4), 25 (Cells 5, 7–8), or 25 (Cell 9) epochs.

**Model EMA:** Exponential Moving Average of model weights (`decay=0.9998`) applied during validation from Cell 7. Best model is saved using EMA weights.

**Model saving:**
- Cells 7–8: Full checkpoint including model, optimizer, scheduler, and EMA state. Saved on best validation Dice.
- Cell 9: Full checkpoint including model, optimizer, scheduler, scaler, hard negative miner state, uncertainty sampler state, and training history. Saved on best combined score subject to a minimum per-class Dice floor. A separate resume checkpoint is written every epoch.

---

## Experiments and Iterations

The ten cells represent a sequential experimental history. Each cell responds to a specific failure mode identified in the prior run.

| Cell | Label | Key Change | Motivation |
|------|-------|-----------|------------|
| 0 | Baseline 2.5D | Single file, 3-level UNet++, CE+Dice | Proof of concept |
| 1 | Swin + multi-file | 4-level UNet++, flat Swin, HD95 metric, 3/1 split | Add transformer component, first real train/val |
| 2 | Memory reduction | Remove Swin, restore checkpointing, reduce filters | OOM errors in Cell 1 |
| 3 | Windowed Swin + val fix | Correct window-partitioned Swin, bilinear val resize | Incorrect Swin in Cell 1; val shape mismatch |
| 4 | Full dataset + AFTL | Directory glob, 80/20 split, AFTL loss, early stopping, per-epoch subsampling | Scale to full dataset; address class imbalance |
| 5 | Attention gates + boundary | 6 attention gates, kornia BoundaryLoss, CompositeLoss | Improve boundary delineation |
| 6 | Diagnostic pipeline | Scan integrity, quick learning test, visualizations | Post-hoc data quality verification |
| 7 | Enhanced 2.5D (L4) | 5-level UNet++, LRU dataset, AdvancedAugmentation, ModelEMA, OneCycleLR | Maximize capacity for dedicated GPU |
| 8 | Bug-fix iteration | Correct decoder ordering, fix window merge, fix GradScaler | NaN losses and runtime errors in Cell 7 |
| 9 | Full 3D + meta-learning | 3D UNet, HNM, UGS, meta-learning, SupCon, topo loss, entropy loss, TTA | Address C3/C4 failure with 3D context and few-shot techniques |

**What changed and why:**

The move from Cell 2 (no Swin) to Cell 3 (windowed Swin) corrects a fundamental implementation error: Cell 1's Swin treated the entire spatial feature map as a single sequence, which is not the intended Swin design and scales quadratically with spatial resolution. The windowed variant in Cell 3 operates on 8×8 local windows, matching the original Swin Transformer paper.

Cell 4 is the first experiment with genuine generalization potential, as prior cells trained on 3–4 scans. The switch from CE+Dice to Asymmetric Focal Tversky Loss is motivated by the extreme foreground rarity — AFTL with `beta=0.7` places 70% of its penalty on false negatives, directly targeting the failure mode of missing small lesions.

Cell 7 represents a hardware upgrade event. Scaling filters from `[16,32,64,128]` to `[64,128,256,512,1024]` and adding the full enhanced component stack is only feasible with 22.5 GB of VRAM. The decoder ordering bug introduced here (each decoder level feeding sequentially into the next rather than computing all same-level nodes before proceeding) violates the UNet++ graph topology and is corrected in Cell 8.

The pivot to 3D in Cell 9 reflects a recognition that the remaining failure modes — particularly poor C3 and C4 recall — are structurally difficult to address with 2.5D processing. Full 3D convolutions capture the volumetric context needed to distinguish enhancing tumor from necrosis, which can appear visually similar in a single axial slice.

[VISUALIZATION_PLACEHOLDER: experiment_comparisons]
Description: Table or parallel coordinate plot comparing key configuration choices across all 10 cells: model dimensionality (2D/2.5D/3D), filter base size, loss function, presence of Swin, attention gates, normalization type, optimizer, and number of training scans. A secondary panel would show the architectural depth progression (number of encoder levels) across cells.

---

## Results

**No quantitative results are recorded in this notebook.**

All training loops print per-epoch metrics to stdout (train loss, validation loss, validation Dice, validation HD metric). However, the notebook contains no preserved cell outputs, no hardcoded result summaries in comments or variables, no saved metric files that are subsequently loaded, and no comparison tables between cells.

The following observations can be inferred from architectural and configuration choices, but they do not constitute measured results:

- The increase in early stopping patience from 7 (Cell 4) to 25 (Cell 5 onward) suggests Cell 4's training converged or plateaued earlier than desired, possibly indicating underfitting or an overly aggressive stopping criterion.
- The filter size oscillation — scaling up in Cell 1, down in Cell 2, up in Cell 5, dramatically up in Cell 7, slightly down in Cell 8 — reflects repeated encounters with GPU memory limits, implying that the target hardware in early cells was insufficient for the intended model capacity.
- The introduction of `ClassAwareHardNegativeMiner` and `UncertaintyGuidedSampler` in Cell 9 strongly implies that C3 and C4 recall remained poor through Cell 8, as these components specifically address the failure to sample and learn from rare-class patches.
- The `ComprehensiveAntiSpeckleLoss` in Cell 9 includes a `slice_consistency` term, suggesting that false positive speckles (isolated single-slice predictions without volumetric coherence) were observed in 2.5D model outputs.

The metric labeled `Val HD95` throughout Cells 1–8 is computed as the symmetric maximum Hausdorff distance (HD100), not the 95th percentile. Any values printed under this label are not comparable to the official BraTS HD95 metric.

[VISUALIZATION_PLACEHOLDER: results_metrics]
Description: Placeholder for training and validation loss curves, per-class Dice scores, and HD95 values across epochs for the final trained model. To be populated upon completion of a full training run with logged outputs.

---

## Key Insights

**2.5D stacking is an effective memory-performance tradeoff for a first pass.** Concatenating adjacent axial slices along the channel dimension provides meaningful 3D context at a fraction of the memory cost of full 3D convolutions, and is a reasonable starting point for constrained hardware.

**Class imbalance cannot be solved by loss weighting alone.** Despite progressively more sophisticated loss functions (CE+Dice → AFTL → CompositeLoss → EnhancedCompositeLoss → ComprehensiveAntiSpeckleLoss), the rare-class failures persisted through Cell 8. The fundamental issue is not how loss is computed but whether rare-class voxels appear in training patches at all. Cell 9's hierarchical sampling addresses this at the data pipeline level, which is where the problem actually lives.

**Gradient checkpointing is essential for deep UNet++ architectures.** UNet++ with dense skip connections retains all intermediate feature maps for the backward pass by default. For a 4-level network with filters `[32,64,128,256]` on 128×128 inputs, this can exceed available GPU memory. Gradient checkpointing at each node — re-running the forward pass during backward — reduces peak activation memory at the cost of approximately 30% additional compute.

**The Swin Transformer implementation requires careful attention to the window partitioning.** The flat-sequence version in Cell 1 is not a Swin Transformer in any meaningful sense and would scale as O(H²W²) attention complexity. The correct windowed implementation in Cell 3 requires padding to ensure divisibility by window size, proper batched window reshaping, and a matching merge operation — a non-trivial implementation detail that introduced a shape error fixed only in Cell 8.

**LRU caching is necessary but not sufficient for multi-worker DataLoader compatibility.** The per-scan caching strategy is correct for single-worker loading but becomes unsafe with `num_workers > 0`, where each worker process holds its own independent cache. The global memory limit `MAX_CACHED_SCANS` is enforced per worker, not globally, meaning actual memory usage scales with worker count.

**Deep supervision annealing is a meaningful training detail.** Gradually reducing the weight of intermediate supervision heads as training progresses allows the network to first learn robust low-level features (supervised by shallow heads) before the final head takes over. The implementation in Cell 9 is more principled than the constant equal-weight deep supervision used in earlier cells.

---

## Limitations

**No quantitative results.** This is the primary limitation. The notebook documents design and implementation but not outcomes. No Dice scores, HD95 values, or combined challenge metrics have been measured and recorded. The project cannot be evaluated as a research contribution in its current state.

**No inference pipeline.** There is no implementation of sliding-window inference for full 3D volumes. All training operates on 2D crops or 3D patches. A challenge submission requires assembling patch-level predictions back into full volumes, which demands careful overlap handling, averaging, and post-processing — none of which is implemented.

**The fifth MRI modality is unidentified.** The notebook uses `IN_CHANNELS=5` with the fifth channel labeled "Grad," but the BraTS-PED challenge provides four standard modalities (T1n, T1c, T2w, T2-FLAIR). If the `.pt` preprocessing created a computed feature (gradient magnitude, edge map, or otherwise) as the fifth channel, this is not documented. If the files actually contain four channels, the `IN_CHANNELS=15` (2.5D) or `IN_CHANNELS=5` (3D) setting is dimensionally incorrect.

**HD95 is incorrectly implemented.** The function `get_hausdorff_score()` and its variants compute the symmetric maximum Hausdorff distance (HD100) but label it `Val HD95`. True HD95 requires the 95th percentile of the asymmetric point-to-set distance distribution. All values printed under the HD95 label in Cells 1–8 are systematically larger than the correct metric and are not comparable to the official BraTS evaluation.

**Label encoding is assumed, not verified.** The mapping of integer labels `{0,1,2,3,4}` to tumor sub-regions (C1–C4) is never explicitly stated or validated. The BraTS native label encoding (`1=NCR, 2=ED, 4=ET`, no label 3) typically requires remapping before use. If the `.pt` preprocessing applied a different remapping than assumed by the notebook, all per-class metrics are measuring the wrong structures.

**Class C2 is absent from all evaluation reporting.** While C2 is included in loss computation (the loops over `range(1, OUT_CHANNELS)` include it), it appears in no logging, no lesion-level metrics, and is not part of the combined score formula. Whether C2 represents a valid distinct structure in BraTS-PED or is merged with another class in the dataset is not clarified.

**The preprocessing pipeline is absent.** The code that converts raw NIfTI MRI volumes to `.pt` tensors does not exist in this notebook. The class weights, intensity scale, and spatial orientation of the data are all dependent on preprocessing decisions that cannot be inspected or reproduced.

**Approximately 65–70% of the notebook is repeated code.** Every cell redefines the same classes and functions from scratch with minor modifications. There is no shared module, no version control between definitions, and no differential documentation explaining what changed.

**Several critical configuration values in Cell 9 are unverified.** The variables `META_K_SHOT`, `PROTO_EPISODE_FREQ`, `ACCUM_STEPS`, `DS_ANNEALING_EPOCH`, `PATCH_SIZE`, `TTA_START`, and `SAMPLING_STRATEGY` are referenced throughout Cell 9 but their definitions were not fully accessible in the notebook's configuration block. In particular, `SAMPLING_STRATEGY` appears to be undefined in the visible code, which would cause a `NameError` at runtime.

---

## Future Work

**Immediate priorities (required before any result is valid):**

1. Identify and document the fifth MRI modality. Verify that all `.pt` files have `data['image'].shape[0] == 5` at dataset initialization and raise an informative error if not.

2. Implement correct HD95 using `scipy.spatial.cKDTree` for point-to-set distance computation followed by `numpy.percentile(..., 95)`. Replace all instances of the current implementation.

3. Document and fix the label encoding. Add a `CLASS_NAMES` constant mapping integer labels to tumor sub-regions and verify the mapping against the official BraTS-PED annotation protocol.

4. Execute a complete, seed-controlled training run of a single architecture (recommended: Cell 8's `EnhancedDenseTrans2D` as the most mature 2.5D model) with all stdout metrics logged to a file. Record at least training loss, validation Dice per class, and corrected HD95 over the full training curve.

**Architectural completion:**

5. Implement sliding-window 3D inference with configurable overlap and Gaussian importance weighting for patch aggregation. This is required to produce full-volume predictions for challenge submission.

6. Resolve the `SAMPLING_STRATEGY` undefined variable in Cell 9. Audit all other configuration constants for completeness before attempting a Cell 9 training run.

7. Fix the `ACCUM_STEPS` asymmetry in Cell 9: divide all auxiliary losses by `ACCUM_STEPS` or document the intended loss scale ratio explicitly.

8. Replace `self.up = lambda ...` with explicit `nn.ModuleList` or named `nn.ConvTranspose2d` attributes across all architecture definitions to avoid potential issues with module registration and serialization.

**Research extensions:**

9. Implement a systematic ablation: hold Cell 8's architecture constant and vary one component at a time (with/without Swin, with/without multi-scale attention gates, AFTL vs. CompositeLoss) against a fixed validation split with recorded metrics.

10. Implement proper HD95. Consider also reporting the normalized surface distance (NSD) metric already implemented in Cell 9 for boundary accuracy evaluation.

11. Extend Cell 9's `UncertaintyGuidedSampler` to use a `WeightedRandomSampler` in the PyTorch DataLoader rather than relying solely on within-scan patch selection weights, enabling uncertainty-driven scan selection across the full dataset.

12. Refactor the notebook into a proper Python module structure: a shared `dataset.py`, `model.py`, `losses.py`, `metrics.py`, and `train.py` with version-controlled definitions, replacing the current copy-paste-per-cell pattern.

---

## Conclusion

This project represents a substantive and technically ambitious effort to develop a pediatric brain tumor segmentation system for the BraTS 2025 Challenge. The experimental arc — from a minimal 2.5D UNet++ through an enhanced windowed-attention architecture to a full 3D network with meta-learning and uncertainty-guided sampling — reflects genuine engagement with the core difficulties of the problem: extreme class imbalance, small rare lesions, and limited training data.

The implementation is sophisticated. The final Cell 9 pipeline in particular incorporates techniques that reflect the current state of the art: k-shot prototype learning for few-shot class recognition, supervised contrastive representation learning, class-aware hard negative mining, and topological regularization. These are not decorative additions — each addresses a specific, documented failure mode.

However, the pipeline is not complete. No quantitative results exist. The inference path from trained model to challenge submission has not been built. Several critical implementation correctness issues (HD95 computation, label mapping, fifth modality identity) remain unresolved. The code requires substantial refactoring for reproducibility.

The next milestone is narrow and concrete: one complete, reproducible training run with recorded validation metrics on a correctly configured version of the Cell 8 architecture. That single run would establish the empirical foundation that the rest of the project currently lacks, and would immediately clarify whether the architectural investments made across ten iterations are translating into segmentation performance.

---

## Appendix

### A. 2.5D Slice Stacking Logic

The core data representation used in Cells 0–8:
```python
target_slice = image_vol[:, slice_idx, :, :]          # shape: (5, H, W)
prev_slice   = image_vol[:, max(0, slice_idx-1), :, :]
next_slice   = image_vol[:, min(D-1, slice_idx+1), :, :]
stacked      = torch.cat([prev_slice, target_slice, next_slice], dim=0)
# stacked shape: (15, H, W)
```

Boundary slices (index 0 and D-1) are handled by repeating the boundary slice rather than zero-padding, preserving realistic intensity values at the volume edges.

### B. Asymmetric Focal Tversky Loss

The loss function used in Cells 4–5, designed for imbalanced foreground segmentation:
```
TI_i = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
AFTL = mean_i[(1 - TI_i)^gamma]
```

With `alpha=0.3`, `beta=0.7`, `gamma=0.75`, and `smooth=1e-6`. The `beta > alpha` configuration places greater penalty on false negatives (missed foreground voxels) than on false positives, which is appropriate when the cost of missing a tumor region exceeds the cost of over-segmentation.

### C. Combined Score Formula (Cell 9)

The checkpoint-saving criterion in Cell 9 is a weighted composite:
```python
def compute_combined_score(class_dice):
    return weighted_combination_of(
        class_dice['WT'], class_dice['TC'],
        class_dice['c1'], class_dice['c3'], class_dice['c4']
    )
```

The exact weights are defined in `compute_combined_score()` in Cell 9. A model is saved only if this score improves **and** `passes_checkpoint_floor()` returns `True` — a minimum per-class Dice requirement preventing degenerate models that score well on WT while failing on rare sub-regions from being checkpointed.

### D. Known Bugs and Status

| Bug | Introduced | Fixed | Status |
|-----|-----------|-------|--------|
| Train == Val dataset | Cell 0 | Cell 1 | Fixed |
| Flat (non-windowed) Swin | Cell 1 | Cell 3 | Fixed |
| Val slices not resized | Cell 1 | Cell 3 | Fixed |
| HD95 labeled as HD100 | Cell 1 | Never | Unfixed |
| Weight decay silently dropped | Cell 5 | Cell 7 | Fixed |
| Decoder level ordering | Cell 7 | Cell 8 | Fixed |
| `_merge_windows` shape error | Cell 7 | Cell 8 | Fixed |
| GradScaler double unscale | Cell 7 | Cell 8 | Fixed |
| `val_pbar` unused | Cell 8 | Never | Unfixed |
| `SAMPLING_STRATEGY` undefined | Cell 9 | Unknown | Unverified |
