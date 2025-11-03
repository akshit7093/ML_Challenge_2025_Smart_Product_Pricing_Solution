# ML Challenge 2025: Smart Product Pricing Solution

**Team Name:** GodParticles  
**Team Members:** Akshit Sharma, Divyansh Rohatgi, Vibhore Sagar, Dilkash Ejaz  
**Submission Date:** 2025-10-13

## Table of Contents
- [Executive Summary](#executive-summary)
- [Methodology](#methodology)
  - [Problem Analysis](#problem-analysis)
  - [Solution Strategy](#solution-strategy)
- [Model Architecture](#model-architecture)
  - [Architecture Overview](#architecture-overview)
  - [Model Components](#model-components)
- [Model Performance](#model-performance)
- [Conclusion & Future Work](#conclusion--future-work)
- [Appendix](#appendix)

---

## Executive Summary

Our project tackles the ML Challenge 2025 by creating a comprehensive pricing model that intelligently synthesizes information from product text and images. We recognized early on that product titles and descriptions were often incomplete, with crucial details like brand, quantity, or key features only visible in the product imagery.

**Core Innovation:** A dual-path visual feature extraction pipeline that treats images as sources of both semantic meaning and explicit text:

- **Semantic Features:** Pre-trained ResNet50 extracts 2048-dimensional vectors representing the image’s “vibe” (colors, shapes, textures).
- **Explicit Text:** A local Visual Language Model (VLM) reads text on packaging (brand, size, quantity) and converts it into clean, machine-readable features.

These visual features are integrated with a robust NLP pipeline that cleans and vectorizes catalog text. The final feature set trains a heavily cross-validated LightGBM model, achieving a strong validation SMAPE score by accurately predicting prices across a wide range of products.

---

## Methodology

### Problem Analysis

Key observations from exploratory data analysis:

- **Price Distribution:** Heavily right-skewed; vast majority of products under \(50 with a long tail of high-ticket items. Applied log1p transformation to tame the distribution.
- **Catalog Content:** Chaotic mix of clean titles, verbose descriptions, HTML tags, and structured nuggets like “Value: 24 Unit: count”. Regex extraction required to unlock numerical gold.
- **Images as Truth:** Titles like “Snack Bars” omit brand, quantity, flavor—details clearly visible on packaging. Text-only models are doomed to fail; images must be read.

### Solution Strategy

Guiding philosophy: **“Leave No Feature Behind.”**

- **Approach Type:** Single heavily feature-engineered LightGBM model.
- **Core Innovation:** Dual-path visual feature extraction:
  - **The “Vibe” (Semantic):** ResNet50 feature vector.
  - **The “Facts” (Explicit):** VLM-generated text describing the image.

---

## Model Architecture

### Architecture Overview

Sequential, multi-stage pipeline. Raw data enters on the left and is progressively enriched through parallel processing streams before unification for the final model.

```
+--------------------------+      +-------------------------------------------+
|      train.csv /         |      |                 image_link                  |
|      test.csv            |      |                                           |
| (catalog_content)        |      +---------------------+---------------------+
+------------+-------------+                            |                     |
             |                            +--------------V-------------+ +-----V------------------+
             |                            |   Image Path 1: The "Facts" | | Image Path 2: The "Vibe" |
+------------V-------------+              | (Local VLM Server)          | | (ResNet50 Feature Extractor) |
|   Text Processing        |              |  - Download Image           | |  - Download Image          |
|  - Regex Value/Unit      |              |  - smolvlm-256m -> Gen. Text| |  - Preprocess (Resize/Norm)|
|  - Text Cleaning         |              |  - Clean Generated Text     | |  - ResNet50 -> 2048D Vector|
|  - TF-IDF Vectorizer     |              |  - TF-IDF Vectorizer        | |  - StandardScaler          |
|   (5000 features)        |              |   (1500 features)           | |   (2048 features)          |
+--------------------------+              +-----------------------------+ +--------------------------+
             |                                          |                             |
             |                                          |                             |
             +------------------+-----------------------+-----------------------------+
                                |
                +---------------V----------------+
                |    Final Feature Combination   |
                | (Horizontally Stack All Sparse |
                |      and Dense Features)       |
                +---------------V----------------+
                                |
                  +-------------V--------------+
                  | LightGBM Regressor Model   |
                  | (20-Fold Cross-Validation) |
                  | (Target: log1p(price))     |
                  +-------------V--------------+
                                |
                       +--------V--------+
                       | Inverse Transform |
                       |  (expm1(pred))  |
                       +--------V--------+
                                |
                      +---------V----------+
                      |  submission.csv    |
                      +--------------------+
```

### Model Components

#### Text Processing Pipeline

- **Structured Data Extraction:** Iterative regex evolution to parse numerical values and units, gracefully handling missing data.
- **Unit Standardization:** Dictionary mapping common variations (‘oz’, ‘ounces’) → ‘ounce’.
- **Text Cleaning:** Strip HTML, remove non-alphabetic characters, lowercase, remove stopwords, lemmatize.
- **Vectorizer:** TfidfVectorizer, max_features=5000, ngram_range=(1, 2).
- **Numerical Features:** Imputed with median, scaled with StandardScaler.
- **Categorical Features:** One-hot encoded.

#### Image Processing Pipeline

**Path 1 – VLM (“Facts”):**

- Download image → local smolvlm-256m-instruct → generate text → clean → TfidfVectorizer (max_features=1500, ngram_range=(1, 2)).
- Engineering: ThreadPoolExecutor for parallel requests, checkpointing every 20 images, repair script for failed links.

**Path 2 – ResNet50 (“Vibe”):**

- Download → resize 256×256 → center-crop 224×224 → ImageNet normalization → pre-trained ResNet50 (fc layer removed) → 2048-D feature vector → StandardScaler.

---

## Model Performance

20-Fold Cross-Validation results (Out-of-Fold predictions):

- **SMAPE:** 16.48 % (mean across 20 folds, σ = 0.21 %)
- **MAE (log-price):** 0.421
- **R² (actual price):** 0.785 (explains ~78.5 % of variance)

Final submission: retrain on 100 % training data.

---

## Conclusion & Future Work

**Lessons Learned:**

- Invest in robust, restartable pipelines—checkpointing and error handling pay off.
- Images are the true source of truth in e-commerce; creative visual feature extraction is a competitive edge.

**Next Steps (with more time):**

- Hyperparameter tuning via Optuna.
- Advanced ensembling & stacking (LightGBM + XGBoost + CatBoost).
- Fine-tune VLM on manually annotated images to reduce hallucinations.

---

## Appendix

### A. Code Artefacts

Complete codebase (notebooks, scripts, processed data): [Google Drive link](https://drive.google.com/your-code-link-here)

Directory layout:

```
/
├── dataset/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   ├── 01_EDA_and_Prototyping.ipynb
│   └── 05_Analysis_and_Visualization.ipynb
├── processed_data/
│   ├── (all generated .npz, .joblib, and .csv files)
└── src/
    ├── 01_preprocess_text.py
    ├── 02_extract_vlm_features.py
    ├── 03_extract_resnet_features.py
    ├── 04_train_model.py
    └── 06_generate_submission.py
```

### B. Feature Importance (Top 5)

| Rank | Feature Name (excerpt)        | Importance (Splits) | Source         |
|------|-------------------------------|----------------------:|----------------|
| 1    | vlm_tfidf_count_…            |                18 452 | Image (VLM)    |
| 2    | num_Value                     |                15 123 | Text (Regex)   |
| 3    | resnet_feature_1024           |                12 890 | Image (ResNet) |
| 4    | text_tfidf_pack_…            |                11 567 | Text (TF-IDF)  |
| 5    | vlm_tfidf_oz_…               |                10 988 | Image (VLM)    |

(Full table available in repo.)