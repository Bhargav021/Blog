---
title: Skills
tags:
  - skills
  - machine-learning
  - skill-map
  - ai-systems
  - ml-engineering
---

## Skill-to-Impact Map

This map ties what I know to what I have shipped, including measurable outcomes.

```mermaid
flowchart LR
  R[Bhargav Limbasia<br/>Applied Data Science, AI Systems, End-to-End ML]

  ML[Machine Learning and Deep Learning]
  NLP[NLP and Computer Vision]
  BE[Backend and Deployment]
  DE[Data Engineering and Pipelines]
  VZ[Visualization and Tools]

  R --> ML
  R --> NLP
  R --> BE
  R --> DE
  R --> VZ

  ML1[3D Medical Imaging]
  ML2[Time Series Forecasting]
  ML3[CNN Architectures]
  ML4[Model Optimization]
  ML --> ML1
  ML --> ML2
  ML --> ML3
  ML --> ML4

  NLP1[LLM Systems]
  NLP2[Prompt Engineering]
  NLP3[Image Processing]
  NLP4[Multimodal Learning]
  NLP --> NLP1
  NLP --> NLP2
  NLP --> NLP3
  NLP --> NLP4

  BE1[API Development]
  BE2[Cloud and DevOps]
  BE3[CI/CD Systems]
  BE --> BE1
  BE --> BE2
  BE --> BE3

  DE1[ETL Pipelines]
  DE2[Data Processing]
  DE3[Distributed Training]
  DE --> DE1
  DE --> DE2
  DE --> DE3

  VZ1[Dashboards and Analytics]
  VZ2[Experiment Tracking]
  VZ3[Developer Tooling]
  VZ --> VZ1
  VZ --> VZ2
  VZ --> VZ3

  ML1 --> T_ML1[PyTorch, MONAI, TorchIO, SimpleITK]
  ML2 --> T_ML2[LSTM, ARIMA, Scikit-Learn]
  ML3 --> T_ML3[SE-ResNet, CUDA]
  ML4 --> T_ML4[AMP, Cosine LR, Checkpointing]

  NLP1 --> T_NLP1[OpenAI GPT, Transformers]
  NLP2 --> T_NLP2[RAG, Prompt Design]
  NLP3 --> T_NLP3[OpenCV, CNNs, Mediapipe]
  NLP4 --> T_NLP4[CLIP, VLA Pipelines]

  BE1 --> T_BE1[Django, ExpressJS, PostgreSQL]
  BE2 --> T_BE2[Azure, AWS, Firebase, Docker]
  BE3 --> T_BE3[Azure DevOps, GitHub Actions, Playwright]

  DE1 --> T_DE1[Pandas, PyMongo]
  DE2 --> T_DE2[Preprocessing Systems, Feature Pipelines]
  DE3 --> T_DE3[GPU Training, DDP Optimization]

  VZ1 --> T_VZ1[D3.js, Tableau, Streamlit, H2O Wave]
  VZ2 --> T_VZ2[Validation Logging, Metric Monitoring]
  VZ3 --> T_VZ3[Git, Jupyter, Reproducible Workflows]

  T_ML1 --> P1[Brain Tumor Segmentation BraTS-PEDs<br/>1000+ MRI scans, +8% Dice, less than 20 min per epoch]
  T_ML2 --> P2[Air Pollution Forecasting<br/>LSTM and ARIMA, RMSE about 15.9, correlation about 0.88]
  T_ML3 --> P3[SE-ResNet Eye Disease Model<br/>96% accuracy, 84k images, IEEE publication]

  T_NLP1 --> P4[Dementia Care GPT Platform Anvayaa<br/>200+ activities, 9 languages, 1600+ videos, 500+ families]
  T_NLP2 --> P5[Visa Query Engine Neurapses<br/>GPT and MongoDB, 100k+ records, +30% query accuracy]

  T_BE1 --> P6[Wearable Data API w4h-api<br/>ExpressJS and PostgreSQL, CI/CD, Playwright testing]
  T_BE2 --> P7[Anvayaa Production System<br/>Docker and Azure DevOps, release time reduced to about 2 hours]

  T_DE1 --> P8[ETL Pipelines Neurapses<br/>100k+ records processed, improved retrieval quality]

  NLP4 --> P9[Diffusion Model for Robotics GLAMOR Lab<br/>ControlNet and Stable Diffusion, multi-view learning, 6-DoF pose prediction]

  classDef root fill:#89b4fa,fill-opacity:0.18,stroke:#89b4fa,color:#cdd6f4,stroke-width:2px;
  classDef ml fill:#89b4fa,fill-opacity:0.18,stroke:#89b4fa,color:#dce8ff;
  classDef nlp fill:#cba6f7,fill-opacity:0.18,stroke:#cba6f7,color:#f0ddff;
  classDef be fill:#a6e3a1,fill-opacity:0.18,stroke:#a6e3a1,color:#dcf8da;
  classDef de fill:#fab387,fill-opacity:0.18,stroke:#fab387,color:#ffe7d0;
  classDef vz fill:#94e2d5,fill-opacity:0.18,stroke:#94e2d5,color:#dcfbf6;
  classDef proj fill:#f9e2af,fill-opacity:0.18,stroke:#f9e2af,color:#fff2d1,stroke-width:2px;

  class R root;
  class ML,ML1,ML2,ML3,ML4,T_ML1,T_ML2,T_ML3,T_ML4 ml;
  class NLP,NLP1,NLP2,NLP3,NLP4,T_NLP1,T_NLP2,T_NLP3,T_NLP4 nlp;
  class BE,BE1,BE2,BE3,T_BE1,T_BE2,T_BE3 be;
  class DE,DE1,DE2,DE3,T_DE1,T_DE2,T_DE3 de;
  class VZ,VZ1,VZ2,VZ3,T_VZ1,T_VZ2,T_VZ3 vz;
  class P1,P2,P3,P4,P5,P6,P7,P8,P9 proj;
```

## Primary Domains

- Machine Learning and Deep Learning
- NLP and Computer Vision
- Backend and Deployment
- Data Engineering and Pipelines
- Visualization and Tools

## Key Design Notes

- Color-coded domains make cluster boundaries obvious in dark mode.
- The ML branch is intentionally denser to reflect strongest depth.
- Outer-ring project nodes include metrics so capability can be verified quickly.

## Project Cards

<h3 id="card-brats">Brain Tumor Segmentation - BraTS-PEDs</h3>

- Problem: pediatric MRI segmentation with class imbalance and noisy labels.
- Approach: TorchIO and MONAI preprocessing, 3D modeling, composite losses, and optimized training.
- Tech: PyTorch, MONAI, TorchIO, SimpleITK.
- Metrics: 1000+ MRI scans, +8% Dice, less than 20 minutes per epoch.
- Link: [BraTs Challenge](https://github.com/Bhargav021/BraTs-Challenge-6)

<h3 id="card-air">Air Pollution Forecasting</h3>

- Problem: non-stationary PM2.5 and PM10 forecasting.
- Approach: time-series feature engineering with hybrid LSTM and ARIMA modeling.
- Tech: LSTM, ARIMA, Scikit-Learn.
- Metrics: RMSE about 15.9 and correlation about 0.88.

<h3 id="card-oct">SE-ResNet Eye Disease Model</h3>

- Problem: automate retinal OCT classification for faster screening.
- Approach: SE-ResNet architecture with augmentation and scheduled optimization.
- Tech: SE-ResNet, CUDA, CNN workflows.
- Metrics: 96% accuracy on 84k images with IEEE publication output.

<h3 id="card-anvayaa-gpt">Dementia Care GPT Platform - Anvayaa</h3>

- Problem: multilingual dementia-care support at scale for families.
- Approach: productized GPT workflows and personalized activity generation.
- Tech: OpenAI GPT, prompt design, backend integration.
- Metrics: 200+ activities, 9 languages, 1600+ videos, adopted by 500+ families.

<h3 id="card-visa">Visa Query Engine - Neurapses</h3>

- Problem: accurate policy search over large immigration datasets.
- Approach: retrieval pipeline plus GPT reasoning with prompt tuning.
- Tech: GPT, MongoDB, RAG, semantic retrieval.
- Metrics: 100k+ records and +30% query accuracy.

<h3 id="card-w4h">Wearable Data API - w4h-api</h3>

- Problem: reliable API delivery for health data workflows.
- Approach: API-first backend with automated test and release workflow.
- Tech: ExpressJS, PostgreSQL, Playwright, CI/CD.

<h3 id="card-anvayaa-release">Anvayaa Production System</h3>

- Problem: slow, manual release process for product updates.
- Approach: deployment automation and containerized delivery pipeline.
- Tech: Docker, Azure DevOps.
- Metrics: release cycle improved from days to about 2 hours.

<h3 id="card-etl">ETL Pipelines - Neurapses</h3>

- Problem: large policy corpus needed clean, queryable structure.
- Approach: automated ETL and validation for retrieval quality.
- Tech: Pandas, PyMongo.
- Metrics: 100k+ records processed.

<h3 id="card-robotics">Diffusion Model for Robotics - GLAMOR Lab</h3>

- Problem: dual-arm manipulation requires richer multimodal action grounding than single-arm systems.
- Approach: action-conditioned training with diffusion and multimodal prompts.
- Tech: ControlNet, Stable Diffusion, VLA pipelines.
- Metrics: multi-view learning and 6-DoF pose prediction experiments.
