# 🛰️ Land Use Land Cover (LULC) Classification Pipeline – Pune Region

## Overview
A modular pipeline for **Land Use Land Cover (LULC)** classification using **multi-band satellite imagery**. Focused on the Pune region, the base version implements a **Random Forest classifier** trained on raw **Surface Reflectance** bands.

> 📌 This project is under active development, with planned improvements for feature engineering, model tuning, and scalability.

---

## Current Status

- Built a basic pipeline for LULC classification.
- Achieved ~98% test accuracy using Random Forest on spectral bands.
- Current outputs include model predictions, evaluation metrics, and prediction maps.

---
## Results on Sentinel-2 L2A SR - Pune Region

<img src="https://github.com/warhazzard/project-LULC-Pune-classification-PIPELINE/blob/main/outputs/classification_2019.png?raw=true">

### Model Evaluation Report - 2019 image

#### **Accuracy:** `0.9872`

---

#### Confusion Matrix

|               | Predicted: Barren | Predicted: Builtup | Predicted: Vegetation | Predicted: Water |
|---------------|------------------:|--------------------:|-----------------------:|------------------:|
| **Actual: Barren**    | 88627 | 181 | 140 | 182 |
| **Actual: Builtup**   | 203   | 19538 | 17  | 0   |
| **Actual: Vegetation**| 114   | 14    | 5593 | 1   |
| **Actual: Water**     | 1004  | 11    | 8    | 30753 |

---

#### Classification Report

| Class        | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| **Barren**     | 0.99      | 0.99   | 0.99     | 89130   |
| **Builtup**    | 0.99      | 0.99   | 0.99     | 19758   |
| **Vegetation** | 0.97      | 0.98   | 0.97     | 5722    |
| **Water**      | 0.99      | 0.97   | 0.98     | 31776   |

---

#### Overall Stats

| Metric         | Score |
|----------------|-------|
| **Accuracy**       | 0.99  |
| **Macro Avg (F1)** | 0.98  |
| **Weighted Avg (F1)** | 0.99  |
| **Total Samples** | 146386 |

---

## 📈 Key Features

- **Random Forest classification** on multi-band rasters.
- **Surface Reflectance** bands used directly as features.
- Basic **train/test split** validation and evaluation.
- **Modular code structure** for easy expansion.

---

## Planned Enhancements

- Add **spatial/contextual features** (texture, edge information).
- **Refine class labels** (e.g., separate agriculture, built-up, barren).
- **Hyperparameter tuning** and explore models like **XGBoost** and **CNNs**.
- Add **modular utilities** (scaling, masking, encoding, CLI options).
- Build a framework for **change detection** over time.
- Improve **reproducibility**: better folder structures, versioning.

