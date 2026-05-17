# 🏥 Medical Diagnosis System

A machine learning–based medical diagnosis system built using Python, Scikit-learn, and FastAPI. This project trains and evaluates predictive models for multiple medical datasets and exposes them through a FastAPI backend for real-time predictions.

📌 Repository: https://github.com/Satyam-Jalvi/Deep-learning-project.git

The system currently supports prediction for:

- Heart Disease Detection  
- Breast Cancer Detection  
- Diabetes Prediction  
- Dermatology Disease Classification  

It demonstrates a complete machine learning pipeline including preprocessing, training, evaluation, visualization, and deployment.

---

# 🚀 Features

## ✅ Multi-Disease Prediction System
Independent ML pipelines for multiple medical conditions.

## 📊 Data Visualization & Analysis
Automatically generates:

- Correlation heatmaps  
- Confusion matrices  
- ROC curves  
- Feature distribution plots  
- Target distribution analysis  

## 🧠 End-to-End ML Pipeline
Covers:

- Data preprocessing  
- Feature engineering  
- Model training  
- Evaluation  
- Model serialization  

## ⚡ FastAPI Backend Integration
Enables real-time prediction via REST APIs.

## 💾 Saved Models
Trained models stored as `.pkl` files for instant reuse without retraining.

---

# 🧠 Tech Stack

- Python  
- Scikit-learn  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- FastAPI  
- Joblib  
- Imbalanced-learn  

---

# 📂 Project Structure

```bash
Deep-learning-project-main/
│
├── app/
│   ├── api.py
│   ├── main.py
│   └── router.py
│
├── data/
│   ├── breast_cancer/
│   ├── dermatology/
│   ├── diabetes/
│   └── heart_disease/
│
├── models/
│   ├── breast_model.pkl
│   ├── dermatology_model.pkl
│   ├── diabetes_model.pkl
│   └── heart_model.pkl
│
├── reports/
│   ├── breast/
│   ├── dermatology/
│   ├── diabetes/
│   └── heart/
│
├── src/
│   ├── config/
│   ├── core/
│   └── visualization/
│
├── main.py
├── predict.py
├── requirements.txt
└── README.md
````

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/Satyam-Jalvi/Deep-learning-project.git
cd Deep-learning-project
```

## 2️⃣ Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Training Pipeline

```bash
python main.py
```

This will:

* Load datasets
* Train models
* Evaluate performance
* Generate reports
* Save trained models inside `models/`

---

# 📊 Outputs Generated

After execution, the system generates:

* Accuracy scores
* Confusion matrices
* ROC curves
* Feature distributions
* Correlation heatmaps

These outputs help interpret model behavior and performance clearly.

---

# 🌐 Run FastAPI Server

```bash
uvicorn app.main:app --reload
```

Server runs at:

[http://127.0.0.1:8000](http://127.0.0.1:8000)

---

# 🔄 Workflow Summary

```text
Dataset → Preprocessing → Training → Evaluation → Model Saving → API Prediction
```

The architecture is modular, making it easy to extend, replace models, or add new diseases.

