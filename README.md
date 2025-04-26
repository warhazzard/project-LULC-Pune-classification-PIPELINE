# 🛰️ Land Use Land Cover (LULC) Classification Pipeline – Pune Region

## Overview
A modular pipeline for **Land Use Land Cover (LULC)** classification using **multi-band satellite imagery**. Focused on the Pune region, the base version implements a **Random Forest classifier** trained on raw **Surface Reflectance** bands.

> 📌 This project is under active development, with planned improvements for feature engineering, model tuning, and scalability.

---

## 🚀 Current Status

- Built a basic pipeline for LULC classification.
- Achieved ~98% test accuracy using Random Forest on spectral bands.
- Current outputs include model predictions, evaluation metrics, and prediction maps.

---

## 📈 Key Features

- **Random Forest classification** on multi-band rasters.
- **Surface Reflectance** bands used directly as features.
- Basic **train/test split** validation and evaluation.
- **Modular code structure** for easy expansion.

---

## 🔧 Planned Enhancements

- Add **spatial/contextual features** (texture, edge information).
- **Refine class labels** (e.g., separate agriculture, built-up, barren).
- **Hyperparameter tuning** and explore models like **XGBoost** and **CNNs**.
- Add **modular utilities** (scaling, masking, encoding, CLI options).
- Build a framework for **change detection** over time.
- Improve **reproducibility**: better folder structures, versioning.

