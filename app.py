import streamlit as st
import pandas as pd
import joblib
import numpy as np
import lightgbm  # LightGBM modelini yükleyebilmek için gerekli

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Diyabet Risk Analizi", layout="wide", page_icon="🏥")

# --- CSS İLE GÖRSELLİK (Referans aldığın projeye benzesin diye) ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        height: 3em;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- MODELİ YÜKLE ---
@st.cache_resource
def load_model():
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'diabetes_model.pkl')
    features_path = os.path.join(current_dir, 'feature_names.pkl')
    
    model = joblib.load(model_path)
    features = joblib.load(features_path)
    return model, features

try:
    model, feature_names = load_model()
    st.sidebar.success("✅ Model başarıyla yüklendi!")
except Exception as e:
    st.error(f"❌ Model dosyaları bulunamadı! Hata: {str(e)}")
    st.info("Lütfen diabetes_model.pkl ve feature_names.pkl dosyalarını app.py ile aynı klasöre koyun.")
    st.stop()

# --- BAŞLIK ---
st.title("🏥 Yapay Zeka Destekli Diyabet Risk Tahmini")
st.markdown("Makine Öğrenmesi (LightGBM) kullanarak diyabet riskinizi saniyeler içinde analiz edin.")
st.markdown("---")

# --- GÖRSELLERİ GÖSTER ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Veri Dağılımı", "🔥 Korelasyon Analizi", "📈 Model Performansı", "🔍 Tahmin Yap"])

with tab1:
    try:
        st.image("indir.png", caption="Hedef Değişken Dağılımı (0: Yok, 1: Var)", use_container_width=True)
    except:
        st.warning("indir.png dosyası bulunamadı")

with tab2:
    try:
        st.image("indir (1).png", caption="Değişkenler Arası Korelasyon Matrisi", use_container_width=True)
    except:
        st.warning("indir (1).png dosyası bulunamadı")

with tab3:
    try:
        st.image("indir (2).png", caption="Model Confusion Matrix", use_container_width=True)
        col_a, col_b = st.columns(2)
        with col_a:
            try:
                st.image("indir (3).png", use_container_width=True)
            except:
                pass
        with col_b:
            try:
                st.image("indir (4).png", use_container_width=True)
            except:
                pass
    except:
        st.warning("Model performans görselleri bulunamadı")

with tab4:
    st.markdown("### 🔬 Diyabet Risk Analizi Formu")
    st.markdown("Aşağıdaki bilgileri doldurup risk analizinizi başlatabilirsiniz.")
    st.markdown("---")

# --- KULLANICI GİRİŞLERİ (FORM) ---
# Ekranı 3 sütuna bölelim
col1, col2, col3 = st.columns(3)

user_input = {}

with col1:
    st.subheader("👤 Kişisel Bilgiler")
    # Yaş kategorik: 1 (18-24) ile 13 (80+) arası
    age_display = st.selectbox("Yaş Grubunuz", 
                 options=range(1, 14), 
                 format_func=lambda x: f"{18 + (x-1)*5}-{24 + (x-1)*5}" if x < 13 else "80+")
    
    # Cinsiyet (0: Kadın, 1: Erkek) - Veri setine göre
    sex = st.radio("Cinsiyet", options=[0, 1], format_func=lambda x: "Kadın" if x==0 else "Erkek")
    
    education = st.slider("Eğitim Seviyesi (1-6)", 1, 6, 4)
    income = st.slider("Gelir Seviyesi (1-8)", 1, 8, 5)

with col2:
    st.subheader("🩺 Sağlık Verileri")
    bmi = st.number_input("Vücut Kitle İndeksi (BMI)", 15.0, 50.0, 25.0)
    gen_hlth = st.slider("Genel Sağlık Durumunuz (1: Mükemmel - 5: Kötü)", 1, 5, 3)
    ment_hlth = st.slider("Son 30 günde ruh sağlığınızın kötü olduğu gün sayısı", 0, 30, 2)
    phys_hlth = st.slider("Son 30 günde fiziksel sağlığınızın kötü olduğu gün sayısı", 0, 30, 2)

with col3:
    st.subheader("⚠️ Risk Faktörleri")
    high_bp = st.checkbox("Yüksek Tansiyonunuz var mı?")
    high_chol = st.checkbox("Yüksek Kolesterolünüz var mı?")
    smoker = st.checkbox("Sigara kullanıyor musunuz? (En az 100 adet)")
    phys_activity = st.checkbox("Düzenli fiziksel aktivite yapıyor musunuz?")
    diff_walk = st.checkbox("Yürürken ciddi zorluk çekiyor musunuz?")

# --- VERİYİ HAZIRLAMA ---
# Kullanıcıdan aldığımız verileri modelin anlayacağı formata (DataFrame) çevirmeliyiz
input_data = pd.DataFrame({
    'HighBP': [1 if high_bp else 0],
    'HighChol': [1 if high_chol else 0],
    'CholCheck': [1], # Varsayılan olarak 1 alıyoruz (arayüzü boğmamak için)
    'BMI': [bmi],
    'Smoker': [1 if smoker else 0],
    'Stroke': [0], # Basitleştirmek için sorulmadı
    'HeartDiseaseorAttack': [0], # Basitleştirmek için sorulmadı
    'PhysActivity': [1 if phys_activity else 0],
    'Fruits': [1], # Varsayılan
    'Veggies': [1], # Varsayılan
    'HvyAlcoholConsump': [0], 
    'AnyHealthcare': [1],
    'NoDocbcCost': [0],
    'GenHlth': [gen_hlth],
    'MentHlth': [ment_hlth],
    'PhysHlth': [phys_hlth],
    'DiffWalk': [1 if diff_walk else 0],
    'Sex': [sex],
    'Age': [age_display],
    'Education': [education],
    'Income': [income]
})

# --- FEATURE ENGINEERING (ÇOK ÖNEMLİ) ---
# Modeli eğitirken yaptığımız türetmeleri burada da yapmak ZORUNDAYIZ!
input_data['Risk_Factor'] = input_data['BMI'] * input_data['HighBP']
input_data['Age_GenHlth'] = input_data['Age'] * input_data['GenHlth']

# Sütun sırasını modelinkiyle aynı yap
input_data = input_data[feature_names]

# --- TAHMİN BUTONU ---
st.markdown("---")
if st.button("RİSK ANALİZİNİ BAŞLAT"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] # Diyabet olma ihtimali
    
    st.write(f"Tahmin Skoru: %{probability*100:.2f}")
    
    if prediction == 1:
        st.error(f"🚨 DİKKAT: Model diyabet riski taşıdığınızı öngörüyor. (Risk: %{probability*100:.1f})")
        st.info("Bu bir tıbbi teşhis değildir. Lütfen en kısa sürede bir sağlık kuruluşuna başvurun.")
    else:
        st.success(f"✅ SONUÇ TEMİZ: Diyabet riski düşük görünüyor. (Risk: %{probability*100:.1f})")