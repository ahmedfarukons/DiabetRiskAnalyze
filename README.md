<div align="center">

# Zero2End ML Bootcamp - Final Project

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.51-red.svg)](https://streamlit.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6-green.svg)](https://lightgbm.readthedocs.io/)
[![MongoDB](https://img.shields.io/badge/MongoDB-7.0-green.svg)](https://www.mongodb.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**This project was developed as a final project for the Zero2End Machine Learning Bootcamp.**

</div>

---

## Demo & Resources

| Resource | Link |
|----------|------|
| YouTube Demo | [Project Introduction Video](https://youtu.be/aPoPtTuUD54) |
| Medium Article | [Diabetes Risk Prediction](https://medium.com/@oahmedfaruk/diyabet-risk-tahmini-24bfa8c5e74b) |
| Presentation | [ML Bootcamp Final Project.pdf](docs/ML%20Bootcamp%20Final%20Proje.pdf) |

---

# Diabetes Risk Analysis Application

## About

A web application that predicts diabetes risk using machine learning. The model, trained with LightGBM algorithm, analyzes users' health data to predict whether they are at risk for diabetes.

## Features

- **LightGBM Model**: High-performance gradient boosting algorithm
- **Interactive Dashboard**: User-friendly interface developed with Streamlit
- **Data Visualization**: Correlation matrix, confusion matrix, and data distribution charts
- **Real-time Prediction**: Instant risk analysis based on user data
- **Responsive Design**: Optimized for all devices
- **Monitoring System**: Prediction logging and alert dashboard with MongoDB
- **Real-time Alerts**: Instant notifications for high-risk predictions

## Installation

### Requirements

- Python 3.8+
- pip
- MongoDB (for monitoring system)

### Steps

1. **Clone the repository:**
```bash
git clone https://github.com/ahmedfarukons/DiabetRiskAnalyze.git
cd DiabetRiskAnalyze
```

2. **Create virtual environment:**
```bash
python -m venv .venv
```

3. **Activate virtual environment:**

Windows:
```bash
.venv\Scripts\activate
```

Linux/Mac:
```bash
source .venv/bin/activate
```

4. **Install dependencies:**
```bash
pip install -r requirements.txt
```

5. **Start MongoDB (for monitoring):**
```bash
# Make sure MongoDB service is running
# Default connection: mongodb://localhost:27017/
```

6. **Run the application:**
```bash
streamlit run app.py
```

7. Browser will automatically open at `http://localhost:8501`

## Dataset

The project uses the **BRFSS (Behavioral Risk Factor Surveillance System) 2015** dataset:

- 21 health indicators
- 253,680 survey responses
- Balanced data distribution (50-50 split)

### Features

- **Demographic**: Age, gender, education, income
- **Health Status**: BMI, general health, mental/physical health
- **Risk Factors**: High blood pressure, cholesterol, smoking, heart disease
- **Lifestyle**: Physical activity, fruit/vegetable consumption, alcohol use

## Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | ~86% |
| F1 Score | ~0.85 |
| Algorithm | LightGBM Classifier |
| Feature Engineering | Risk_Factor, Age_GenHlth |

## Monitoring System

The project includes a MongoDB-based monitoring system:

- **Prediction Logging**: Every prediction is automatically saved to database
- **Alert Dashboard**: High-risk predictions (>60%) are tracked in real-time
- **Statistics**: Daily/weekly/total alert counts
- **Risk Distribution**: Low/Medium/High risk visualization
- **Trend Analysis**: Last 7 days alert chart

### Database Schema
```javascript
{
  "prediction_id": "uuid",
  "timestamp": "datetime",
  "input_features": { "BMI": 25, "Age": 5, ... },
  "risk_score": 0.72,
  "risk_level": "high",
  "is_alert": true
}
```

## Project Structure

```
DiabetRiskAnalyze/
├── app.py                    # Streamlit application
├── database.py               # MongoDB connection module
├── diabetes_model.pkl        # Trained model
├── feature_names.pkl         # Feature names
├── requirements.txt          # Python dependencies
├── README.md                 # Documentation
├── archive/                  # Datasets
├── images/                   # Visual assets
├── docs/                     # Documentation files
└── kaggle_dataset_download.ipynb
```

## Application Tabs

| Tab | Description |
|-----|-------------|
| Prediction | Risk analysis form and results |
| Data Analysis | Target variable distribution and correlation analysis |
| Model Performance | Confusion matrix, ROC curve, and metrics |
| Alert Monitoring | Real-time high-risk tracking dashboard |

## Usage

1. Go to the **"Prediction"** tab
2. Fill in all form fields (personal info, health data, risk factors)
3. Click **"START RISK ANALYSIS"** button
4. View your results
5. Track all predictions from **"Alert Monitoring"** tab

**Important**: This application is for informational purposes only and does not replace medical diagnosis.

## Technologies

- Python 3.13
- Streamlit (Web interface)
- LightGBM (ML model)
- MongoDB (Prediction logging)
- PyMongo (MongoDB driver)
- Scikit-learn (Model evaluation)
- Pandas, NumPy, Joblib

## License

This project was developed for educational purposes.

## Developer

**Ahmed Faruk**
- GitHub: [@ahmedfarukons](https://github.com/ahmedfarukons)
- Medium: [@oahmedfaruk](https://medium.com/@oahmedfaruk)

## Acknowledgements

- Kaggle - For the dataset
- BRFSS - For survey data
- Zero2End Bootcamp Team - For training and mentorship

---

If you liked this project, don't forget to give it a star!
