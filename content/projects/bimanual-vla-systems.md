---
title: Bimanual Robotic Manipulation with Vision-Language-Action Systems
tags:
  - projects
  - robotics
  - diffusion
---

- Tagline: Training diffusion models for dual-arm manipulation
- Category: Current Research
- GitHub: [spacefly](https://github.com/iamty8/spacefly)

## Context

Most VLA models are designed for single-arm manipulation. Bimanual tasks need dataset conversion and action representation that preserve dual-gripper coordination.

## Role

Research Engineer owning dataset conversion, action heatmap generation, ControlNet+GenIMA training, and Hugging Face dataset compatibility fixes.

## Work completed

- Converted PerAct2 bimanual dataset (700+ demonstrations) to LLaVa prompt format
- Generated dual-gripper action heatmaps:
  - left gripper: blue/cyan
  - right gripper: orange/red
  - explicit open/closed states
- Trained ControlNet with GenIMA on Stable Diffusion 1.5 using image and action conditioning
- Fixed iterable dataset compatibility issues for large trajectory training
- Improved validation with configurable guidance scales and trajectory-level logging

## Impact and learning

- Successfully trained with dual-gripper action conditioning at scale
- Learned to bridge robotics action-vector datasets with modern prompt-driven VLA pipelines
