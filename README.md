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

## Kaynaklar

| Kaynak | Link |
|--------|------|
| YouTube | [Proje Videosu](https://youtu.be/aPoPtTuUD54) |
| Medium | [Diyabet Risk Tahmini](https://medium.com/@oahmedfaruk/diyabet-risk-tahmini-24bfa8c5e74b) |
| Sunum | [ML Bootcamp Final Proje.pdf](docs/ML%20Bootcamp%20Final%20Proje.pdf) |

---

# EN

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

```bash
git clone https://github.com/ahmedfarukons/DiabetRiskAnalyze.git
cd DiabetRiskAnalyze
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
streamlit run app.py
```

### Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | ~86% |
| F1 Score | ~0.85 |
| Algorithm | LightGBM |

---

# TR

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

```bash
git clone https://github.com/ahmedfarukons/DiabetRiskAnalyze.git
cd DiabetRiskAnalyze
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
streamlit run app.py
```

### Model Performansı

| Metrik | Değer |
|--------|-------|
| Doğruluk | ~%86 |
| F1 Skoru | ~0.85 |
| Algoritma | LightGBM |

---

## Monitoring Sistemi

MongoDB tabanlı tahmin takip sistemi:

- Tahmin loglama
- Yüksek risk (>%60) için alert dashboard
- Günlük/haftalık istatistikler
- Risk dağılımı görselleştirmesi

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

## Proje Yapısı

```
DiabetRiskAnalyze/
├── app.py                    # Streamlit uygulaması
├── database.py               # MongoDB modülü
├── diabetes_model.pkl        # Eğitilmiş model
├── feature_names.pkl         # Feature isimleri
├── requirements.txt          # Bağımlılıklar
├── README.md                 # Dokümantasyon
├── archive/                  # Veri setleri
├── images/                   # Görseller
└── docs/                     # Dokümanlar
```

---

## Teknolojiler

Python, Streamlit, LightGBM, MongoDB, PyMongo, Scikit-learn, Pandas, NumPy

---

## Geliştirici

**Ahmed Faruk**
- GitHub: [@ahmedfarukons](https://github.com/ahmedfarukons)
- Medium: [@oahmedfaruk](https://medium.com/@oahmedfaruk)

## Teşekkürler

- Kaggle - Veri seti
- BRFSS - Anket verileri
- Zero2End Bootcamp Ekibi

---

<div align="center">

Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!

</div>
