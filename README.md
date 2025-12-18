# Human Heat Flux Prediction from Chair Sensors

This project was developed as part of the **CS-433: Machine Learning** course in collaboration with the **ICE Lab**.  
The objective is to predict **human heat flux (thigh and back)** from **chair pressure sensor data**, using classical machine learning models and deep learning approaches.

---

## Project Overview

The dataset is composed of:
- Pressure sensors embedded in a chair
- Time-series measurements
- Participant metadata (sex, height, weight, etc.)

The task consists in predicting:
- **Thigh heat flux**
- **Back heat flux**

Models are evaluated using **subject-wise splits** to avoid data leakage.

---

## Repository Structure

### 1. `Data_preprocessing.ipynb`
- Parses and aggregates raw data provided by the ICE Lab
- Synchronizes sensor data, heat flux measurements, and timestamps
- Produces cleaned train/test CSV files

### 2. `Validation_subjectwise.ipynb`
- Defines the subject-wise validation strategy
- Justifies the choice of train / validation / test splits
- Ensures no participant leakage between splits

### 3. `Baseline_training.ipynb`
- Implements baseline regression models:
  - Linear Regression
  - Ridge
  - Lasso
- Serves as a performance reference

### 4. `Personal_pressure_models.ipynb`
- Adds personal characteristics (sex, height, weight, etc.)
- Implements a **binary posture classifier** (seated vs standing)
- Filters unreliable samples when the participant is not seated

### 5. `xgboost_benchmark.ipynb`
- Explores non-linear models using **XGBoost**
- Performs subject-wise cross-validation and hyperparameter search
- Compares performance with linear baselines

### 6. `lstm_benchmark.ipynb`
- Implements a **temporal LSTM model**
- Uses sliding windows over sensor time-series
- Captures temporal dynamics of pressure and heat flux

---

## Authors

Emma Berenholt
Zacharie Bourlard
Fanny Missillier

Project developed for CS-433 — EPFL  
In collaboration with the ICE Lab
