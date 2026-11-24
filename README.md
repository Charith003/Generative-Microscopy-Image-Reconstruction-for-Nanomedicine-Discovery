# NanoScope – Generative Microscopy Image Reconstruction for Nanomedicine Discovery

<p align="center">
  <img src="https://img.shields.io/badge/AI-Deep%20Learning-blueviolet?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Computer-Vision-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Domain-Nanomedicine-00A884?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Frontend-React%20%2B%20TypeScript-61DAFB?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Deployment-Vercel-black?style=for-the-badge" />
</p>

<p align="center">
  <b>An AI-assisted microscopy image reconstruction and nanomedicine research platform.</b>
</p>

<p align="center">
  Enhance microscopy images, explore cellular morphology, characterize nanoparticles, evaluate formulation properties, and investigate drug candidate screening workflows.
</p>

<p align="center">
  <a href="https://nano-vision-ai.vercel.app">Live Application</a> •
  <a href="https://github.com/Charith003/Generative-Microscopy-Image-Reconstruction-for-Nanomedicine-Discovery">GitHub Repository</a> •
  <a href="https://github.com/Charith003/nu_data">Dataset Repository</a>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Project Objectives](#project-objectives)
- [System Architecture](#system-architecture)
- [Deep Learning Model](#deep-learning-model)
- [Research Methodology](#research-methodology)
- [Core Features](#core-features)
- [Nanomedicine Analysis Modules](#nanomedicine-analysis-modules)
- [Technology Stack](#technology-stack)
- [Dataset](#dataset)
- [Evaluation Metrics](#evaluation-metrics)
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [Google Colab Notebooks](#google-colab-notebooks)
- [Live Deployment](#live-deployment)
- [Limitations](#limitations)
- [Future Enhancements](#future-enhancements)
- [Applications](#applications)
- [Author](#author)
- [License](#license)

---

## Overview

**NanoScope (NanoVision AI)** is a generative microscopy image reconstruction and nanomedicine discovery research project that explores the application of artificial intelligence, deep learning, and computer vision to biomedical image analysis.

The research focuses on improving the visual quality of microscopy images, extracting meaningful cellular and nanoparticle characteristics, and developing computational workflows for evaluating nanomedicine candidates.

The deep-learning component explores **NanoVisionNet-X**, a multi-task architecture incorporating a ResNet18-based encoder and multiple analytical prediction heads. The reconstruction experiments additionally explore a Residual U-Net denoising architecture using a combination of L1 and SSIM loss.

Alongside the research notebooks, the project provides an interactive React-based web application with image-upload, analysis visualization, screening, optimization, and drug-prediction modules.

### Project Highlights

- Generative microscopy image reconstruction using deep learning.
- Residual U-Net-based image denoising experiments.
- ResNet18-based multi-task NanoVisionNet-X architecture.
- Cellular morphology and nuclei analysis.
- Image reconstruction quality evaluation.
- Nanoparticle characterization and aggregation analysis.
- Formulation stability and nano-bio interaction indicators.
- Multi-factor screening and risk scoring.
- Drug candidate prediction and molecular property exploration.
- Interactive web dashboard with persistent browser-based analysis history.

---

## Problem Statement

Microscopy images are essential for studying cellular structures, nuclei, and nanoparticle behavior. However, image acquisition processes can introduce noise, blur, and structural distortions that make accurate visual interpretation and computational analysis more difficult.

Traditional image-processing methods may not adequately preserve fine structural details while removing unwanted noise.

Additionally, evaluating nanomedicine candidates involves several interconnected properties, including particle morphology, stability, aggregation, biological interactions, transport behavior, and potential toxicity.

This project explores an integrated computational approach that combines microscopy image reconstruction, feature extraction, multi-task learning, and analytical screening workflows.

The objective is to create a research-oriented platform that supports image enhancement and the exploration of nanomedicine-related analytical indicators.

---

## Project Objectives

1. Investigate deep-learning approaches for reconstructing microscopy images affected by noise and degradation.
2. Explore U-Net-based denoising architectures with structural similarity-aware loss functions.
3. Develop a ResNet18-based multi-task framework for extracting and analyzing image representations.
4. Study cellular morphology, nuclei characteristics, and nanoparticle distribution.
5. Evaluate reconstruction performance using PSNR and SSIM.
6. Explore nano-bio interaction and formulation-related analytical indicators.
7. Develop multi-factor screening and candidate-ranking workflows.
8. Build an interactive web application for visualizing analysis outputs.
9. Provide reproducible research experiments through Google Colab notebooks.

---

## System Architecture

The project consists of a research/model-development layer and a React-based visualization application.

### Overall Workflow

```text
               MICROSCOPY IMAGE
                       |
                       v
              IMAGE PREPROCESSING
                       |
                       v
             IMAGE QUALITY ANALYSIS
                       |
                       v
             DEEP LEARNING RESEARCH
                       |
             +---------+---------+
             |                   |
             v                   v
        RESIDUAL U-NET       RESNET18
        RECONSTRUCTION       ENCODER
             |                   |
             v                   v
       DENOISED IMAGE       FEATURE EXTRACTION
                                 |
                                 v
                         MULTI-TASK MODULES
                                 |
              +------------------+------------------+
              |                  |                  |
              v                  v                  v
          MORPHOLOGY         FORMULATION        NANO-BIO
           ANALYSIS           ANALYSIS          ANALYSIS
              |                  |                  |
              +------------------+------------------+
                                 |
                                 v
                         SCREENING MODULE
                                 |
                                 v
                     DRUG CANDIDATE ANALYSIS
                                 |
                                 v
                       RESULT VISUALIZATION
                                 |
                                 v
                      WEB APPLICATION
```

*The diagram describes the intended research workflow. The current web interface uses a separate mock-analysis engine for its displayed analysis metrics; it is not yet a live integration with the trained neural networks.*

---

## Deep Learning Model

### 1. NanoVisionNet-X

NanoVisionNet-X is the proposed multi-task deep-learning architecture explored in the research notebooks.

It uses a **ResNet18-based encoder** to extract visual features from microscopy images. The extracted representation is passed into multiple specialized heads.

#### Architecture Components

| Component | Purpose |
|---|---|
| ResNet18 Encoder | Visual feature extraction |
| Reconstruction Decoder | Image reconstruction |
| Morphology Head | Morphological feature estimation |
| Formulation Head | Formulation-related property estimation |
| Nano-Bio Head | Biological interaction indicators |
| Screening Head | Candidate screening indicators |

The architecture is designed to explore multiple related analytical objectives using shared image representations.

### 2. Residual U-Net Reconstruction

The separate image-reconstruction experiment uses a Residual U-Net-style denoising architecture.

Key components:

- Encoder-decoder architecture.
- Skip connections for preserving spatial information.
- Residual learning for image reconstruction.
- Grayscale microscopy image processing.
- Image degradation experiments using noise and blur-like corruption.

#### Training Configuration

| Parameter | Configuration |
|---|---|
| Model | Residual U-Net |
| Framework | TensorFlow / Keras |
| Input Size | 128 × 128 × 1 |
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Epochs | 12 |
| Loss Function | L1 + SSIM |
| Dataset | Nuclei microscopy images |

#### Composite Loss Function

The reconstruction experiment uses:

\[
L = 0.7L_{L1} + 0.3L_{SSIM}
\]

Where:

- \(L_{L1}\) measures pixel-wise reconstruction error.
- \(L_{SSIM}\) measures structural similarity loss.

This combination is intended to balance pixel-level accuracy and structural preservation.

---

## Research Methodology

The research workflow is organized into the following stages.

### Stage 1 – Dataset Preparation

- Load microscopy images and corresponding annotations.
- Inspect the dataset structure.
- Resize images to the required input dimensions.
- Normalize image pixel values.
- Prepare training and testing samples.
- Generate degraded image inputs for denoising experiments.

### Stage 2 – Image Reconstruction

- Build the Residual U-Net architecture.
- Generate reconstructed microscopy images.
- Apply composite reconstruction loss.
- Train using the prepared image samples.
- Evaluate reconstructed outputs.

### Stage 3 – Feature Extraction and Multi-Task Learning

- Implement the ResNet18 encoder.
- Extract image representations.
- Explore specialized prediction heads.
- Analyze morphology, formulation, nano-bio interaction, and screening-related outputs.

### Stage 4 – Quantitative Evaluation

Evaluate reconstructed images using:

- Peak Signal-to-Noise Ratio (PSNR).
- Structural Similarity Index (SSIM).

Explore segmentation-related evaluation using Dice and Intersection over Union (IoU).

### Stage 5 – Analytical Screening

Explore multi-factor indicators associated with nanoparticle morphology, formulation properties, aggregation, stability, and biological interactions.

### Stage 6 – Visualization and Application

Present analysis outputs using an interactive web interface with charts, history management, optimization workflows, and screening dashboards.

---

## Core Features

### 1. Microscopy Image Upload and Analysis

- Upload microscopy images.
- Preview selected images.
- Generate an analysis result.
- Display analytical metrics and visualizations.
- Save analysis records to browser history.

### 2. Image Reconstruction Research

- Residual U-Net-based reconstruction experiments.
- Noise removal and denoising.
- Structural preservation using SSIM-aware loss.
- Reconstruction quality evaluation.
- Optional sharpening post-processing in the research notebook.

### 3. Morphological Characterization

- Nuclei count.
- Mean particle area.
- Area variation.
- Circularity estimation.
- Particle size distribution.
- Density distribution.
- Aggregation indicators.

### 4. Drug Screening Dashboard

- Select historical analysis samples.
- Search and filter samples.
- Compare risk, final screening score, and PSNR.
- Explore individual screening sections.
- Review analytical indicators.

### 5. Optimization Module

- Select saved analysis samples.
- Compare original and optimized metrics.
- Explore lower-risk score transformations.
- Generate an image-enhanced output.
- Save optimized results.
- Download optimized images.

### 6. Drug Prediction Module

- Select original or optimized analysis records.
- Enter molecular and biological parameters.
- Explore SMILES-based compound information.
- Estimate efficacy and toxicity indicators.
- Calculate multi-criteria screening scores.
- Save prediction records.
- Export and import analysis data.

### 7. Analysis History

- Store analysis records in browser local storage.
- View historical analysis entries.
- Open individual records.
- Delete selected records.
- Preserve original and optimized results.

---

## Nanomedicine Analysis Modules

The application organizes nanomedicine-related analytical workflows into multiple modules.

| Module | Analytical Focus |
|---|---|
| Image Reconstruction | Denoising and structural preservation |
| Morphology Analysis | Nuclei count, area, circularity |
| Formulation Analysis | Stability and transport indicators |
| Nano-Bio Interaction | Membrane interaction and cytotoxicity indicators |
| Aggregation Analysis | Clustering and particle overlap |
| Drug Prediction | Molecular properties and efficacy indicators |
| Candidate Screening | Risk scoring and multi-factor classification |
| Optimization | Comparative score and image enhancement |

**Note:** These analytical modules represent computational research indicators and exploratory predictions. They should not be interpreted as experimentally validated biological measurements or clinical recommendations.

---

## Technology Stack

### Deep Learning and Research

| Technology | Application |
|---|---|
| Python | Research and experimentation |
| PyTorch | NanoVisionNet-X architecture experiments |
| TensorFlow / Keras | Residual U-Net reconstruction |
| NumPy | Numerical computation |
| OpenCV | Image processing |
| Scikit-image | PSNR and SSIM evaluation |
| Matplotlib | Research visualization |
| Pandas | Data analysis |
| Scikit-learn | Machine-learning utilities |

### Frontend Development

| Technology | Application |
|---|---|
| React.js | User interface |
| TypeScript | Type-safe application development |
| Vite | Development and build tooling |
| Tailwind CSS | Styling |
| shadcn/ui | Interface components |
| Radix UI | Accessible UI primitives |
| Framer Motion | Animations |
| Recharts | Data visualization |
| React Router | Navigation |
| TanStack Query | Query-state management |
| Lucide React | Icons |
| Vitest | Testing |

### Deployment

- Vercel
- GitHub
- Google Colab

---

## Dataset

The research uses nuclei microscopy data and related image annotations.

**Primary dataset repository:**

[Charith003/nu_data](https://github.com/Charith003/nu_data)

The image reconstruction notebook references the `nuclei_partial_annotations` dataset directory.

The older and experimental notebooks also include references to nuclei dataset structures such as `stage_1_train`.

### Dataset Preparation

The research workflow involves:

- Loading microscopy images.
- Reading available annotations and masks.
- Resizing images.
- Normalizing input data.
- Creating training and testing partitions.
- Generating degraded inputs for reconstruction.

Dataset availability, licensing, and usage conditions should be verified from the original dataset sources.

---

## Evaluation Metrics

### Image Reconstruction Metrics

**1. Peak Signal-to-Noise Ratio (PSNR)**

Measures the reconstruction error relative to the maximum possible pixel intensity.

\[
PSNR = 10\log_{10}\left(\frac{MAX_I^2}{MSE}\right)
\]

A higher PSNR generally indicates lower pixel-level reconstruction error.

**2. Structural Similarity Index (SSIM)**

Measures structural similarity between the original and reconstructed images using luminance, contrast, and structural information.

Values closer to 1 indicate greater structural similarity.

### Segmentation Metrics

**3. Dice Similarity Coefficient**

\[
Dice = \frac{2|A\cap B|}{|A|+|B|}
\]

Measures the overlap between predicted and ground-truth segmentation masks.

**4. Intersection over Union (IoU)**

\[
IoU = \frac{|A\cap B|}{|A\cup B|}
\]

Measures the intersection relative to the union of predicted and ground-truth masks.

*Dice and IoU are included as research evaluation indicators. Their values must be calculated against actual ground-truth annotations to constitute validated segmentation performance.*

---

## Project Structure

```text
Generative-Microscopy-Image-Reconstruction-for-Nanomedicine-Discovery/
│
├── notebooks/
│   ├── Minor_model.ipynb
│   ├── Minor_old_model.ipynb
│   ├── Minor_preprocessing.ipynb
│   │
│   ├── NanoVision_Net_X_Colab.ipynb
│   ├── NanoVision_Image_Reconstruction_Colab.ipynb
│   ├── NanoVision_Optimization_Colab.ipynb
│   ├── NanoVision_Upload_Analyze_Colab.ipynb
│   ├── NanoVision_Drug_Discovery_Analysis_Colab.ipynb
│   └── NanoVision_Drug_Screening_Dashboard_Colab.ipynb
│
├── public/
│   ├── favicon.ico
│   ├── placeholder.svg
│   └── robots.txt
│
├── src/
│   ├── assets/
│   ├── components/
│   │   ├── ui/
│   │   ├── AppNavbar.tsx
│   │   ├── ImageUploader.tsx
│   │   ├── NavLink.tsx
│   │   └── StatCard.tsx
│   │
│   ├── hooks/
│   ├── lib/
│   │   ├── drugPredictionDb.ts
│   │   ├── historyDb.ts
│   │   ├── mockAnalysis.ts
│   │   └── optimizer.ts
│   │
│   ├── pages/
│   │   ├── Analyze.tsx
│   │   ├── DrugPrediction.tsx
│   │   ├── HistoryDetailPage.tsx
│   │   ├── HistoryPage.tsx
│   │   ├── Index.tsx
│   │   ├── OptimizationPage.tsx
│   │   ├── Screening.tsx
│   │   └── NotFound.tsx
│   │
│   ├── App.tsx
│   ├── App.css
│   ├── index.css
│   └── main.tsx
│
├── index.html
├── package.json
├── package-lock.json
├── tailwind.config.ts
├── tsconfig.json
├── vite.config.ts
├── vitest.config.ts
└── README.md
```

---

## Installation and Setup

### Prerequisites

- Node.js (LTS recommended).
- npm.
- Git.
- Python environment for research notebooks.
- Google Colab for GPU-assisted notebook experiments, where available.

### 1. Clone the Repository

```bash
git clone https://github.com/Charith003/Generative-Microscopy-Image-Reconstruction-for-Nanomedicine-Discovery.git
```

### 2. Navigate to the Project

```bash
cd Generative-Microscopy-Image-Reconstruction-for-Nanomedicine-Discovery
```

### 3. Install Dependencies

```bash
npm install
```

### 4. Start Development Server

```bash
npm run dev
```

### 5. Open in Browser

```text
http://localhost:5173
```

### 6. Build for Production

```bash
npm run build
```

### 7. Run Tests

```bash
npm test
```

### 8. Run Linting

```bash
npm run lint
```

---

## Google Colab Notebooks

The repository includes dedicated notebooks for individual research and application workflows.

| Notebook | Purpose |
|---|---|
| Minor_model.ipynb | PyTorch ResNet18-based NanoVisionNet-X architecture experiments |
| Minor_old_model.ipynb | Earlier autoencoder, GAN, and multi-task model experiments |
| Minor_preprocessing.ipynb | Dataset preparation and preprocessing experiments |
| NanoVision_Net_X_Colab.ipynb | NanoVisionNet-X demonstration and analytical outputs |
| NanoVision_Image_Reconstruction_Colab.ipynb | Residual U-Net microscopy reconstruction |
| NanoVision_Optimization_Colab.ipynb | Optimization workflow demonstration |
| NanoVision_Upload_Analyze_Colab.ipynb | Image upload and analysis workflow |
| NanoVision_Drug_Discovery_Analysis_Colab.ipynb | Drug discovery and molecular prediction workflow |
| NanoVision_Drug_Screening_Dashboard_Colab.ipynb | Multi-sample screening dashboard |

### Run a Notebook

1. Open the notebook from the repository.
2. Select **Open in Colab** or upload the `.ipynb` file into Google Colab.
3. Select an available runtime.
4. Execute the cells in sequence.
5. Follow the notebook instructions for dataset preparation and input files.

---

## Live Deployment

The React-based NanoVision application is deployed on Vercel.

**Live URL:**

https://nano-vision-ai.vercel.app

The web application provides access to the image analysis, screening dashboard, history management, optimization, and drug prediction interfaces.

---

## Applications

Potential research applications include:

- Microscopy image enhancement.
- Cellular and nuclei image analysis.
- Nanoparticle morphology characterization.
- Formulation stability research.
- Nano-bio interaction analysis.
- Computational drug candidate screening.
- Biomedical image processing research.
- AI-assisted nanomedicine investigations.

The system is intended to support research exploration and computational analysis, not replace laboratory experiments or professional medical assessment.

---

## Author

**Charith Mandhalapu**

B.Tech – Computer Science and Engineering  
Specialization: Artificial Intelligence and Machine Learning  
SRM Institute of Science and Technology, Chennai

### Connect

- GitHub: [@Charith003](https://github.com/Charith003)
- Project Repository: [NanoScope – Generative Microscopy Image Reconstruction](https://github.com/Charith003/Generative-Microscopy-Image-Reconstruction-for-Nanomedicine-Discovery)
- Live Application: [NanoVision AI](https://nano-vision-ai.vercel.app)

---

## License

A license has not yet been specified for this repository. Please check the repository for licensing updates before reusing or distributing the code.

---

<p align="center">
  <b>NanoScope – Exploring AI-Driven Microscopy Reconstruction and Nanomedicine Discovery.</b>
</p>
