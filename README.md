ICE Human Heat Flux Prediction

Chair Pressure Sensing for Back and Thigh Heat Flux Estimation

Overview

This project investigates the prediction of human back and thigh heat flux using chair-embedded pressure sensors, complemented with personal characteristics and posture information.
The work is conducted using real experimental data provided by the ICE Lab, and explores both classical machine learning models and sequence-based deep learning approaches.

The main objectives are:
	•	To build robust regression models for back_flux and thigh_flux
	•	To assess the impact of pressure sensors, personal characteristics, and posture
	•	To compare linear, tree-based, and temporal (LSTM) models under strict subject-wise validation

Repository Structure:

ICE-human-heatflux-project/
│
├── Cleaned_Datasets/
│   ├── Train_final.csv
│   ├── Test_final.csv
│   ├── Train_final_with_posture.csv
│   └── Test_final_with_posture.csv
│
├── runs/
│   └── Saved trained models, bundles, and results
│
├── utils/
│   ├── Baseline_training_utils.py
│   ├── Data_extraction_utils.py
│   ├── Model_utils.py
│   └── Validation_subjectwise_utils.py
│
├── Data_preprocessing.ipynb
├── Validation_subjectwise.ipynb
├── Baseline_training.ipynb
├── Personal_Pressure_models.ipynb
├── XGBoost_benchmark.ipynb
├── LSTM_benchmark.ipynb
│
└── README.md
