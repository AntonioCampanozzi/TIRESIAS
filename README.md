# TIRESIAS: TImely REsistance Surmise In Additive manufacturing Samples

## Overview
**TIRESIAS** is a machine learning framework designed to predict the mechanical resistance of Additive Manufacturing (AM) specimens early in their testing phase. The primary objective is to reduce operational costs, accelerate experimental workflows, and prevent the destruction of complex prototypes by creating a virtual proxy of physical tests. 

## Objective & Earliness
The core concept of this project is **earliness** . The goal is to provide reliable outputs using only a partial sequence (prefix) of the available experimental data. The project aims to accurately predict the maximum stress supported by a specimen prior to the dynamic failure threshold characteristic of its respective cluster.

## Dataset
*   **Source:** Publicly available dataset from the Harvard Dataverse.
*   **Composition:** 360 individual `.TXT` files, each representing a discrete experimental run in the form of time series.
*   **Test Modalities:** 180 three-point bending tests and 180 compression tests.
*   **Materials:** Thermoplastics including Polyamide (PA), Polycarbonate (PC), Polyethylene Terephthalate Glycol (PETG), and Polylactic Acid (PLA).
*   **Configurations:** Tested across three specific wall thicknesses (1 mm, 2 mm, and 3 mm).

## Methodology
The project structure follows the CRISP-DM methodology across multiple modeling iterations:

### 1. Data Preprocessing & Clustering
*   Data extraction transformed raw experimental files into structured Pandas DataFrames.
*   **Unsupervised Learning:** K-Medoids clustering combined with a Dynamic Time Warping (DTW) distance metric was used to identify natural patterns and isolate experiments with distinct mechanical trajectories without relying solely on static categorical labels.

### 2. Modeling Iterations
*   **Iteration 1 (LSTM):** A Long Short-Term Memory deep learning network was initially used on zero-padded sequential prefixes. While effective for full-sequence regression, it failed to provide acceptable early-warning detection
*   **Iteration 2 (XGBoost):** The approach shifted to tabular machine learning. Temporal prefixes were summarized into statistical descriptors including Mean, Standard Deviation, Skewness, Kurtosis, Last observed value, and Delta. Extreme Gradient Boosting (XGBoost) significantly outperformed the LSTM, achieving strong early prediction capabilities.
*   **Iteration 3 (Dynamic Thresholds):** Re-evaluated the performance using dynamic failure thresholds based on cluster medoids. For bending tests, structural failure corresponds to the peak force of the curve, whereas for compression, it corresponds to the first distinct peak.

## Key Results
*   The results proved that a streamlined methodology based on statistical feature engineering (XGBoost) surpasses complex deep learning architectures (LSTM) in both predictive accuracy and operational manageability.
*   **Bending Tests:** The XGBoost model proved highly robust, consistently delivering accurate predictions well in advance of the structural failure point (which typically occurs at approx. 85% of the sequence), on average at 44% of the series.
*   **Compression Tests:** Predicting compression failure proved more challenging as it occurs much earlier (approx. 33% of the series), though the global XGBoost model still maintained a strong overall performance with positive predictive lead times, with a delta from the threshold of 12%, corresponding to the 20% of the series.

