# ICU Mortality Risk Predictor

## Project Overview
A clinical decision-support tool designed to predict in-hospital mortality for ICU patients using vital signs and lab results from the first 24 hours of admission.

This project prioritizes **Patient Safety (Recall)** over aggregate accuracy. It uses a **Class-Weighted Logistic Regression** pipeline to ensure high-risk patients are not missed.

## Live Demo
(https://icu-mortality-prediction.streamlit.app)

## Model Performance
* **Recall (Sensitivity):** ~73% (Optimized to detect ~3 out of 4 mortality cases)
* **Precision:** ~25% (Accepts higher false positive rate to function as an effective screening tool)
* **Key Features:** Lactate, GCS Total, Ventilation Status, Systolic BP.

## Tech Stack
* **Python** (Pandas, Scikit-Learn)
* **Streamlit** (Web App Interface)
* **Altair** (Data Visualization)

## Project Structure
* `app.py`: The main Streamlit dashboard application.
* `logreg_pipeline.pkl`: The trained ML pipeline (StandardScaler + LogisticRegression).
* `Awibi_project_notebook.ipynb`: Exploration, training, and evaluation code.
