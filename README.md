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

#### **Accuracy:** `0.9820`

---

#### Confusion Matrix

|                   | Predicted: Barren | Predicted: Built-up | Predicted: Vegetation | Predicted: Water |
|-------------------|------------------:|--------------------:|----------------------:|-----------------:|
| **Actual: Barren**     | 1950 | 9    | 21   | 20   |
| **Actual: Built-up**   | 9    | 1985 | 6    | 0    |
| **Actual: Vegetation** | 3    | 1    | 1996 | 0    |
| **Actual: Water**      | 73   | 1    | 1    | 1925 |

---

#### Classification Report

| Class       | Precision | Recall | F1-Score | Support |
|:-----------:|:---------:|:------:|:--------:|:-------:|
| Barren      | 0.96      | 0.97   | 0.97     | 2000    |
| Built-up    | 0.99      | 0.99   | 0.99     | 2000    |
| Vegetation  | 0.99      | 1.00   | 0.99     | 2000    |
| Water       | 0.99      | 0.96   | 0.98     | 2000    |

---

#### Overall Stats

- **Overall Accuracy:** 0.9820
- **Macro Average F1-Score:** 0.98
- **Weighted Average F1-Score:** 0.98
- **Total Samples:** 8000

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

