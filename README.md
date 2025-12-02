<div align="center">

# 🎓 Zero2End ML Bootcamp - Final Projesi

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.51-red.svg)](https://streamlit.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6-green.svg)](https://lightgbm.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 🚀 **Bu proje, Zero2End Machine Learning Bootcamp'i kapsamında final projesi olarak geliştirilmiştir.**

</div>

---

# 🏥 Diyabet Risk Analizi Uygulaması

## 📋 Proje Hakkında

Bu proje, makine öğrenmesi kullanarak diyabet riskini tahmin eden bir web uygulamasıdır. LightGBM algoritması ile eğitilmiş model, kullanıcıların sağlık verilerini analiz ederek diyabet riski taşıyıp taşımadıklarını tahmin eder.

## 🎯 Özellikler

- 🤖 **LightGBM Modeli**: Yüksek performanslı gradient boosting algoritması
- 📊 **İnteraktif Dashboard**: Streamlit ile geliştirilmiş kullanıcı dostu arayüz
- 📈 **Veri Görselleştirme**: Korelasyon matrisi, confusion matrix ve veri dağılımı grafikleri
- 🔍 **Anlık Tahmin**: Kullanıcı verilerine göre gerçek zamanlı risk analizi
- 📱 **Responsive Tasarım**: Her cihazda mükemmel görünüm

## 🚀 Kurulum

### Gereksinimler

- Python 3.8+
- pip

### Adımlar

1. **Repository'yi klonlayın:**
```bash
git clone https://github.com/ahmedfarukons/DiabetRiskAnalyze.git
cd DiabetRiskAnalyze
```

2. **Virtual environment oluşturun:**
```bash
python -m venv .venv
```

3. **Virtual environment'ı aktif edin:**

Windows:
```bash
.venv\Scripts\activate
```

Linux/Mac:
```bash
source .venv/bin/activate
```

4. **Gerekli kütüphaneleri yükleyin:**
```bash
pip install -r requirements.txt
```

5. **Uygulamayı başlatın:**
```bash
streamlit run app.py
```

6. Tarayıcınızda otomatik olarak `http://localhost:8501` adresi açılacaktır.

## 📊 Veri Seti

Proje, **BRFSS (Behavioral Risk Factor Surveillance System) 2015** veri setini kullanmaktadır. Bu veri seti şu özellikleri içerir:

- 21 sağlık göstergesi
- 253,680 anket yanıtı
- Dengeli veri dağılımı (50-50 split)

### Özellikler

- **Demografik**: Yaş, cinsiyet, eğitim, gelir
- **Sağlık Durumu**: BMI, genel sağlık, mental/fiziksel sağlık
- **Risk Faktörleri**: Yüksek tansiyon, kolesterol, sigara, kalp hastalığı
- **Yaşam Tarzı**: Fiziksel aktivite, meyve/sebze tüketimi, alkol kullanımı

## 🧠 Model Performansı

- **Doğruluk (Accuracy)**: ~86%
- **F1 Score**: ~0.85
- **Algoritma**: LightGBM Classifier
- **Feature Engineering**: Risk_Factor, Age_GenHlth

## 📁 Proje Yapısı

```
DiabetRiskAnalyze/
│
├── app.py                          # Streamlit uygulaması
├── diabetes_model.pkl              # Eğitilmiş model
├── feature_names.pkl               # Feature isimleri
├── requirements.txt                # Python bağımlılıkları
├── README.md                       # Proje dokümantasyonu
│
├── archive/                        # Veri setleri
│   ├── diabetes_012_health_indicators_BRFSS2015.csv
│   ├── diabetes_binary_5050split_health_indicators_BRFSS2015.csv
│   └── diabetes_binary_health_indicators_BRFSS2015.csv
│
├── images/                         # Görsel dosyaları
│   ├── diabetes_distribution.png  # Veri dağılımı
│   ├── correlation_matrix.png     # Korelasyon matrisi
│   ├── confusion_matrix.png       # Confusion matrix
│   ├── model_metrics_1.png        # Model metrikleri
│   └── model_metrics_2.png        # Model metrikleri
│
├── docs/                           # Dokümantasyon
│   └── ML Bootcamp Final Proje.pdf
│
└── kaggle_dataset_download.ipynb   # Veri indirme notebook
```

## 🎨 Uygulama Ekran Görüntüleri

### Ana Sayfa
- Diyabet Risk Tahmini formu
- Kişisel bilgiler (yaş, cinsiyet, eğitim, gelir)
- Sağlık verileri (BMI, genel sağlık durumu)
- Risk faktörleri (tansiyon, kolesterol, sigara)

### Veri Analizi Sekmeleri
1. **📊 Veri Dağılımı**: Hedef değişkenin dağılımı
2. **🔥 Korelasyon Analizi**: Değişkenler arası ilişkiler
3. **📈 Model Performansı**: Confusion matrix ve metrikler
4. **🔍 Tahmin Yap**: Risk analiz formu

## 💡 Kullanım

1. Uygulamayı başlattıktan sonra **"🔍 Tahmin Yap"** sekmesine gidin
2. Formdaki tüm alanları doldurun:
   - Kişisel bilgileriniz
   - Sağlık verileriniz
   - Risk faktörleriniz
3. **"RİSK ANALİZİNİ BAŞLAT"** butonuna tıklayın
4. Sonuçlarınızı görüntüleyin

⚠️ **Önemli**: Bu uygulama sadece bilgilendirme amaçlıdır ve tıbbi teşhis yerine geçmez.

## 🛠️ Teknolojiler

- **Python 3.13**
- **Streamlit**: Web arayüzü
- **LightGBM**: Makine öğrenmesi modeli
- **Scikit-learn**: Model değerlendirme ve preprocessing
- **Pandas**: Veri manipülasyonu
- **NumPy**: Sayısal işlemler
- **Joblib**: Model serileştirme

## 📝 Lisans

Bu proje eğitim amaçlı geliştirilmiştir.

## 👨‍💻 Geliştirici

**Ahmed Faruk**
- GitHub: [@ahmedfarukons](https://github.com/ahmedfarukons)
- Repository: [DiabetRiskAnalyze](https://github.com/ahmedfarukons/DiabetRiskAnalyze)

## 🙏 Teşekkürler

- Kaggle - Veri seti için
- BRFSS - Anket verileri için
- ML Bootcamp - Eğitim ve mentorluk için

---

⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!

