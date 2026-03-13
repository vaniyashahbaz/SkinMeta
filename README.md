# SkinMeta 
### Explainable AI-Based Personalized Skincare Recommendation System

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![CSharp](https://img.shields.io/badge/C%23-Windows%20Forms-purple?logo=dotnet)
![ML](https://img.shields.io/badge/ML-Random%20Forest%20%2B%20CNN-green)
![XAI](https://img.shields.io/badge/XAI-SHAP-orange)
![Status](https://img.shields.io/badge/Status-Phase%202%20Complete-brightgreen)

---

## What is SkinMeta?

SkinMeta is a desktop-based AI-powered skincare recommendation system built with **C# Windows Forms** and a **Python ML backend**. It analyses a user's skin type, acne condition (detected from a photo via CNN), environmental conditions, hormonal phase, and allergen profile to recommend a personalised skincare treatment — with a clear, human-readable explanation for every recommendation using **SHAP (Explainable AI)**.

---

## The Problem

Most skincare apps give the same generic advice to everyone. They don't consider:
- Your specific acne *type* (blackheads vs cystic are treated very differently)
- Your local humidity and pollution levels
- Your hormonal cycle phase
- Your ingredient allergies
- Your skin sensitivity to products

SkinMeta combines all of these factors using machine learning instead of hardcoded rules.

---

## How It Works

```
User uploads photo → CNN detects acne type
User fills questionnaire → 6 questions (skin type, severity, location, sensitivity, allergen, concern)
App fetches weather → OpenWeatherMap API (humidity, pollution, temperature)
App calculates → Hormonal phase from last period date
         │
         ▼
   Random Forest ML Model
   (12 features → predicts treatment category)
         │
         ▼
   SQL Database query → returns matching branded products
         │
         ▼
   SHAP Explanation → "Why this recommendation?"
```

---

## Project Architecture

```
SkinMeta/
├── SkinMeta_App/              ← C# Windows Forms application (SQL Server backend)
│   ├── Forms/                 ← All UI forms (Login, Questionnaire, Recommendation, etc.)
│   ├── Services/              ← SkinRecommendationService.cs (calls Flask API)
│   └── DatabaseHelper.cs      ← SQL Server connection helper
│
├── data_scripts/              ← Phase 2: All ML data pipeline scripts
│   ├── generate_dataset_final.py   ← Generates 1,500 synthetic user profiles
│   ├── preprocess_final.py         ← Cleaning, encoding, scaling, EDA charts
│   └── data/
│       ├── raw/               ← Raw dataset with noise (do not commit large files)
│       ├── processed/         ← Cleaned, encoded, scaled CSVs + JSON encoders
│       └── eda_charts/        ← 10 EDA visualization PNG files
│
├── ml_model/                  ← Trained model artifacts (Phase 3 — coming soon)
│   ├── model.pkl              ← Trained Random Forest classifier
│   ├── scaler.pkl             ← MinMaxScaler
│   └── label_encoders.json    ← Category-to-number mappings
│
├── cnn_model/                 ← CNN for acne image detection (Phase 3 — coming soon)
│   └── cnn_model.h5           ← Trained CNN weights
│
├── flask_api/                 ← Python Flask REST API (Phase 4 — coming soon)
│   └── app.py                 ← Prediction endpoint called by C# app
│
├── report/
│   └── SkinMeta_Technical_Report.pdf   ← Phase 1 + Phase 2 LaTeX report
│
└── README.md
```

---

## AI Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Acne Detection | CNN (Convolutional Neural Network) | Classifies uploaded photo into 5 acne types |
| Treatment Recommendation | Random Forest Classifier | Predicts treatment category from 12 features |
| Explainability | SHAP (SHapley Additive exPlanations) | Explains why each recommendation was made |

---

## The 12 Features

| # | Feature | Source |
|---|---------|--------|
| 1 | `skin_type` | Questionnaire Q1 |
| 2 | `acne_type` | CNN image detection |
| 3 | `acne_severity` | Questionnaire Q2 |
| 4 | `breakout_location` | Questionnaire Q3 |
| 5 | `product_sensitivity` | Questionnaire Q4 |
| 6 | `allergen` | Questionnaire Q5 |
| 7 | `skin_concern` | Questionnaire Q6 |
| 8 | `humidity` | OpenWeatherMap API |
| 9 | `pollution` | OpenWeatherMap API |
| 10 | `temperature` | OpenWeatherMap API |
| 11 | `age_group` | User registration |
| 12 | `hormonal_phase` | Calculated from last period date |

---

## Dataset

### Acne Image Dataset (CNN)
- **Source:** Kaggle — `tiswan14/acne-dataset-image`
- **Size:** 4,617 images across 5 classes
- **Classes:** Blackheads, Cyst, Papules, Pustules, Whiteheads
- **Split:** 70% train / 15% validation / 15% test

### User Profile Dataset (Recommendation Model)
- **Type:** Synthetic — generated using clinical dermatology rules
- **Size:** 1,500 records (1,530 raw with noise)
- **Features:** 12 input features + 1 target variable
- **Target:** 8 treatment categories

> **Why synthetic data?** SkinMeta is a new system with no existing users. Real patient skincare data is private and unavailable. Synthetic data generation using dermatological rules is standard practice in healthcare AI research.

---

## Phase Status

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Problem Definition | ✅ Complete |
| Phase 2 | Data Acquisition & Preprocessing | ✅ Complete |
| Phase 3 | Model Training & Evaluation | 🔄 In Progress |
| Phase 4 | Flask API + C# Integration | ⏳ Upcoming |
| Phase 5 | Testing & Deployment | ⏳ Upcoming |

---

## Running Phase 2 Scripts

### Requirements
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Steps
```bash
# 1. Generate the synthetic dataset
python data_scripts/generate_dataset_final.py

# 2. Clean, encode, scale, and generate EDA charts
python data_scripts/preprocess_final.py
```

Output files will appear in `data_scripts/data/`.

> **Tip:** You can also run these scripts in Google Colab — just upload the two `.py` files and run them as cells using `!python filename.py`

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Desktop Application | C# Windows Forms (.NET) |
| Database | SQL Server (DESKTOP-D41EMMQ / SkinMeta) |
| ML Backend | Python 3.9+ |
| ML Models | scikit-learn (Random Forest), TensorFlow/Keras (CNN) |
| Explainability | SHAP |
| Weather API | OpenWeatherMap |
| ML API Bridge | Flask REST API |
| Report | LaTeX |

---

## Team

> **Note:** Commit history must reflect active participation from all group members. Each member should commit their own work directly — do not batch-commit on behalf of others.

---

## License

This project was developed for academic purposes as part of a Software Engineering course.
