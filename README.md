<div align="center">

# Zero2End ML Bootcamp - Final Project

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.51-red.svg)](https://streamlit.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6-green.svg)](https://lightgbm.readthedocs.io/)
[![MongoDB](https://img.shields.io/badge/MongoDB-7.0-green.svg)](https://www.mongodb.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**This project was developed as a final project for the Zero2End Machine Learning Bootcamp.**

---

### Demo Video

[![Diabetes Risk Analysis Demo](https://img.youtube.com/vi/aPoPtTuUD54/maxresdefault.jpg)](https://youtu.be/aPoPtTuUD54)

*Click the image to watch the demo video*

</div>

---

## Resources / Kaynaklar

| Resource | Link |
|----------|------|
| YouTube Demo | [Project Video](https://youtu.be/aPoPtTuUD54) |
| Medium Article | [Diabetes Risk Prediction](https://medium.com/@oahmedfaruk/diyabet-risk-tahmini-24bfa8c5e74b) |
| Presentation | [ML Bootcamp Final Project.pdf](docs/ML%20Bootcamp%20Final%20Proje.pdf) |

---

# English

## Diabetes Risk Analysis Application

### About

A web application that predicts diabetes risk using machine learning. The model, trained with LightGBM algorithm, analyzes users' health data to predict whether they are at risk for diabetes.

### Features

- **LightGBM Model**: High-performance gradient boosting algorithm
- **Interactive Dashboard**: User-friendly interface developed with Streamlit
- **Data Visualization**: Correlation matrix, confusion matrix, and data distribution charts
- **Real-time Prediction**: Instant risk analysis based on user data
- **Monitoring System**: Prediction logging and alert dashboard with MongoDB
- **Real-time Alerts**: Instant notifications for high-risk predictions

### Installation

#### Requirements
- Python 3.8+
- MongoDB (for monitoring system)

#### Steps

```bash
# Clone repository
git clone https://github.com/ahmedfarukons/DiabetRiskAnalyze.git
cd DiabetRiskAnalyze

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

Browser will open at `http://localhost:8501`

### Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | ~86% |
| F1 Score | ~0.85 |
| Algorithm | LightGBM Classifier |

### Dataset

**BRFSS 2015** - 253,680 survey responses with 21 health indicators.

---

# Türkçe

## Diyabet Risk Analizi Uygulaması

### Hakkında

Makine öğrenmesi kullanarak diyabet riskini tahmin eden bir web uygulaması. LightGBM algoritması ile eğitilmiş model, kullanıcıların sağlık verilerini analiz ederek diyabet riski taşıyıp taşımadıklarını tahmin eder.

### Özellikler

- **LightGBM Modeli**: Yüksek performanslı gradient boosting algoritması
- **İnteraktif Dashboard**: Streamlit ile geliştirilmiş kullanıcı dostu arayüz
- **Veri Görselleştirme**: Korelasyon matrisi, confusion matrix ve veri dağılımı grafikleri
- **Anlık Tahmin**: Kullanıcı verilerine göre gerçek zamanlı risk analizi
- **Monitoring Sistemi**: MongoDB ile tahmin loglama ve alert dashboard
- **Gerçek Zamanlı Alertler**: Yüksek riskli tahminler için anlık bildirimler

### Kurulum

#### Gereksinimler
- Python 3.8+
- MongoDB (monitoring sistemi için)

#### Adımlar

```bash
# Repository'yi klonla
git clone https://github.com/ahmedfarukons/DiabetRiskAnalyze.git
cd DiabetRiskAnalyze

# Virtual environment oluştur ve aktif et
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Bağımlılıkları yükle
pip install -r requirements.txt

# Uygulamayı başlat
streamlit run app.py
```

Tarayıcı otomatik olarak `http://localhost:8501` adresinde açılacaktır.

### Model Performansı

| Metrik | Değer |
|--------|-------|
| Doğruluk | ~%86 |
| F1 Skoru | ~0.85 |
| Algoritma | LightGBM Classifier |

### Veri Seti

**BRFSS 2015** - 21 sağlık göstergesi içeren 253,680 anket yanıtı.

---

## Monitoring System / Monitoring Sistemi

MongoDB-based monitoring system for tracking predictions:

- Prediction logging / Tahmin loglama
- Alert dashboard for high-risk (>60%) / Yüksek risk için alert dashboard
- Daily/weekly statistics / Günlük/haftalık istatistikler
- Risk distribution visualization / Risk dağılımı görselleştirmesi

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

---

## Project Structure / Proje Yapısı

```
DiabetRiskAnalyze/
├── app.py                    # Streamlit application
├── database.py               # MongoDB module
├── diabetes_model.pkl        # Trained model
├── feature_names.pkl         # Feature names
├── requirements.txt          # Dependencies
├── README.md                 # Documentation
├── archive/                  # Datasets
├── images/                   # Visual assets
└── docs/                     # Documentation files
```

---

## Technologies / Teknolojiler

Python, Streamlit, LightGBM, MongoDB, PyMongo, Scikit-learn, Pandas, NumPy

---

## Developer / Geliştirici

**Ahmed Faruk**
- GitHub: [@ahmedfarukons](https://github.com/ahmedfarukons)
- Medium: [@oahmedfaruk](https://medium.com/@oahmedfaruk)

## Acknowledgements / Teşekkürler

- Kaggle - Dataset / Veri seti
- BRFSS - Survey data / Anket verileri
- Zero2End Bootcamp Team / Zero2End Bootcamp Ekibi

---

<div align="center">

If you liked this project, don't forget to give it a star!

Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!

</div>
