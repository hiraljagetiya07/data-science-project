# IoT DDoS Attack Detection System

A modular machine learning pipeline designed to detect and classify DDoS attacks (specifically targeting MQTT and UDP protocols) in IoT environments. This repository implements a full data science workflow from raw data cleaning to advanced model evaluation.

## 📌 Project Overview
As IoT devices become ubiquitous, they are increasingly vulnerable to Distributed Denial of Service (DDoS) attacks. This project utilizes the **UL-ECE-IoT2025** dataset to:
* **Preprocess** network traffic by handling missing values and outliers.
* **Analyze** data through Exploratory Data Analysis (EDA) and dimensionality reduction.
* **Detect** malicious activity using both supervised and unsupervised learning techniques.

---

## 📂 Repository Structure
```text
├── data/
│   ├── raw/                # Original CSV datasets (MQTT/UDP)
│   └── processed/          # Cleaned CSVs ready for modeling
├── outputs/                # Auto-generated results & visualizations
│   ├── figures/            # EDA plots (Target dist, Heatmaps, Boxplots)
│   ├── clustering/         # K-Means, DBSCAN, and Hierarchical results
│   ├── pca/                # PCA projections and variance plots
│   ├── svd/                # SVD components and analysis
│   └── models/             # Performance metrics, ROC curves, and summaries
├── data_cleaning.py        # Pipeline for noise removal and IQR outlier clipping
├── graph_plotting.py       # EDA script for comprehensive visualizations
├── pca_analysis.py         # Dimensionality reduction via Principal Component Analysis
├── svd_analysis.py         # Latent feature extraction via SVD
├── clustering.py           # Unsupervised grouping (K-Means, DBSCAN, etc.)
├── logistic_regression.py   # Statistical classification model
├── naive_bayes.py          # Probabilistic classification model
├── random_forest.py        # Ensemble-based classification model
└── metrices_summary.py     # Global model performance comparison logic
