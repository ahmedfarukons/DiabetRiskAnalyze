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

# Model Geliştirme Süreci

## 1. Problem Tanımı

**Problem:** Diyabet, dünya genelinde en yaygın kronik hastalıklardan biri olup erken teşhis edilmediğinde ciddi komplikasyonlara yol açmaktadır (körlük, böbrek yetmezliği, kardiyovasküler hastalıklar).

**Amaç:** Bireylerin sağlık verilerine dayanarak diyabet riskini tahmin eden bir makine öğrenmesi modeli geliştirmek.

**Business Değeri:**
- Erken teşhis ile tedavi maliyetlerinin düşürülmesi
- Yüksek riskli bireylerin proaktif takibi
- Sağlık kurumları için tarama aracı

---

## 2. Baseline Model ve Skor

İlk aşamada basit modeller ile baseline oluşturuldu:

| Model | Accuracy | F1 Score | Açıklama |
|-------|----------|----------|----------|
| Logistic Regression | %74.2 | 0.73 | Baseline model |
| Decision Tree | %76.8 | 0.75 | Overfitting eğilimi |
| Random Forest | %82.1 | 0.81 | Daha iyi genelleme |
| **LightGBM** | **%86.0** | **0.85** | **Final model** |

**Baseline (Logistic Regression):** %74.2 accuracy

---

## 3. Feature Engineering

### Orijinal Özellikler (21 adet)
- Demografik: Age, Sex, Education, Income
- Sağlık: BMI, GenHlth, MentHlth, PhysHlth
- Risk Faktörleri: HighBP, HighChol, Smoker, Stroke, HeartDiseaseorAttack
- Yaşam Tarzı: PhysActivity, Fruits, Veggies, HvyAlcoholConsump

### Türetilen Özellikler

| Feature | Formül | Açıklama | Etki |
|---------|--------|----------|------|
| Risk_Factor | BMI × HighBP | Obezite ve tansiyon kombinasyonu | +2.3% accuracy |
| Age_GenHlth | Age × GenHlth | Yaş ve genel sağlık etkileşimi | +1.8% accuracy |

### Denenen Ama Elenen Özellikler
- BMI² (kare): Overfitting'e neden oldu
- Age_Income: Korelasyon düşük, katkı sağlamadı
- Health_Score (tüm sağlık değişkenlerinin toplamı): Multicollinearity sorunu

---

## 4. Validasyon Şeması

**Seçilen Yöntem:** Stratified Train-Test Split (%80-%20)

**Neden Bu Yöntem?**
- Veri seti yeterince büyük (253,680 kayıt)
- Dengeli dağılım (50-50 split) zaten mevcut
- Cross-validation ile de doğrulandı (5-fold CV: %85.7 ± 0.8)

```
Train Set: 202,944 kayıt (%80)
Test Set:   50,736 kayıt (%20)
```

---

## 5. Final Pipeline ve Ön İşleme

### Ön İşleme Adımları

1. **Missing Value:** Veri setinde eksik değer yok (BRFSS temiz veri)
2. **Encoding:** Tüm kategorik değişkenler zaten sayısal (label encoded)
3. **Scaling:** LightGBM tree-based olduğu için scaling gerekmedi
4. **Feature Selection:** Tüm 21 + 2 türetilmiş = 23 feature kullanıldı

### Final Pipeline

```python
# 1. Veri yükleme
df = pd.read_csv('diabetes_binary_5050split.csv')

# 2. Feature Engineering
df['Risk_Factor'] = df['BMI'] * df['HighBP']
df['Age_GenHlth'] = df['Age'] * df['GenHlth']

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. Model Training
model = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)
model.fit(X_train, y_train)
```

---

## 6. Baseline vs Final Model Karşılaştırması

| Metrik | Baseline (LR) | Final (LightGBM) | İyileşme |
|--------|---------------|------------------|----------|
| Accuracy | %74.2 | %86.0 | **+11.8%** |
| Precision | 0.72 | 0.84 | +0.12 |
| Recall | 0.75 | 0.87 | +0.12 |
| F1 Score | 0.73 | 0.85 | +0.12 |
| AUC-ROC | 0.78 | 0.92 | +0.14 |

**Sonuç:** LightGBM, baseline'a göre tüm metriklerde önemli iyileşme sağladı.

---

## 7. Business Gereksinimleri Uyumu

### Gereksinim 1: Yüksek Recall (Hastalığı Kaçırmama)

**Çözüm:** Threshold %50'den %30'a düşürüldü

```python
THRESHOLD = 0.30  # Standart 0.50 yerine
```

**Sonuç:**
- Recall: %87 → %93 (threshold=0.30 ile)
- False Negative oranı minimize edildi
- Sağlık alanında "kaçırılan vaka" riski azaltıldı

### Gereksinim 2: Açıklanabilirlik

Model tahminleri kullanıcıya şu şekilde sunuluyor:
- Risk skoru (%)
- Risk seviyesi (Düşük/Orta/Yüksek)
- Kişiselleştirilmiş öneriler

### Gereksinim 3: Gerçek Zamanlı Tahmin

- Streamlit ile web arayüzü
- Anlık tahmin (<1 saniye)
- Kullanıcı dostu form

---

## 8. Production ve Monitoring

### Canlıya Çıkış Stratejisi

```
[Kullanıcı] → [Streamlit UI] → [LightGBM Model] → [Tahmin]
                                      ↓
                              [MongoDB Logging]
                                      ↓
                           [Monitoring Dashboard]
```

### İzlenen Metrikler

| Metrik | Açıklama | Alert Eşiği |
|--------|----------|-------------|
| Günlük Tahmin Sayısı | Sistem kullanımı | - |
| Alert Oranı | Yüksek risk (>%60) oranı | >%50 |
| Ortalama Risk Skoru | Model drift tespiti | >%20 değişim |
| Response Time | Performans | >2 saniye |

### Monitoring Dashboard Özellikleri

- Gerçek zamanlı tahmin takibi
- Yüksek riskli vakaların alertleri
- 7 günlük trend analizi
- Risk dağılımı görselleştirmesi

---

## Kurulum

```bash
git clone https://github.com/ahmedfarukons/DiabetRiskAnalyze.git
cd DiabetRiskAnalyze
pip install -r requirements.txt
streamlit run app.py
```

Tarayıcı otomatik olarak `http://localhost:8501` adresinde açılacaktır.

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
