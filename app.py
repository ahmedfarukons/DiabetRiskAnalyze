import streamlit as st
import pandas as pd
import joblib
import numpy as np
import lightgbm  # LightGBM modelini yükleyebilmek için gerekli
from datetime import datetime

# Monitoring Database
import database as db

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
st.title("🏥 Diyabet Risk Tahmini")
st.markdown("Makine Öğrenmesi (LightGBM) kullanarak diyabet riskinizi saniyeler içinde analiz edin.")
st.markdown("---")

# --- GÖRSELLERİ GÖSTER ---
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Tahmin Yap", "📊 Veri Analizi", "📈 Model Performansı", "🚨 Alert Monitoring"])

with tab1:
    st.markdown("### 🔬 Diyabet Risk Analizi Formu")
    st.markdown("Aşağıdaki bilgileri doldurup risk analizinizi başlatabilirsiniz.")
    
    # BMI Hesaplayıcı - Sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📏 BMI Hesaplayıcı")
        st.info("Vücut Kitle İndeksinizi bilmiyorsanız buradan hesaplayabilirsiniz.")
        
        height_cm = st.number_input("Boy (cm)", min_value=100, max_value=250, value=170, step=1)
        weight_kg = st.number_input("Kilo (kg)", min_value=30, max_value=300, value=70, step=1)
        
        if st.button("🧮 BMI Hesapla", use_container_width=True):
            height_m = height_cm / 100
            calculated_bmi = weight_kg / (height_m ** 2)
            st.success(f"**BMI'niz:** {calculated_bmi:.1f}")
            
            # BMI kategorisi
            if calculated_bmi < 18.5:
                st.info("📊 Kategori: Zayıf")
            elif calculated_bmi < 25:
                st.success("📊 Kategori: Normal")
            elif calculated_bmi < 30:
                st.warning("📊 Kategori: Fazla Kilolu")
            else:
                st.error("📊 Kategori: Obez")
            
            st.caption(f"💡 Form'da BMI olarak **{calculated_bmi:.1f}** kullanabilirsiniz.")
    
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
    
    # Eğitim seviyesi (1-6 arası)
    education = st.slider("🎓 Eğitim Seviyesi (1: En Düşük - 6: En Yüksek)", 1, 6, 4)
    
    # Gelir seviyesi (1-8 arası)
    income = st.slider("💰 Gelir Seviyesi (1: En Düşük - 8: En Yüksek)", 1, 8, 5)

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

# --- VERİ ANALİZİ SEKMESİ ---
with tab2:
    st.markdown("### 📊 Veri Seti Analizi")
    st.markdown("BRFSS 2015 veri setinin detaylı analizi")
    
    with st.expander("📈 Hedef Değişken Dağılımı", expanded=False):
        try:
            st.image("images/diabetes_distribution.png", 
                    caption="Diyabet durumunun dağılımı (0: Yok, 1: Var)", 
                    use_container_width=True)
            st.info("Veri seti dengeli bir şekilde dağıtılmıştır (50-50 split).")
        except:
            st.warning("Veri dağılımı görseli bulunamadı")
    
    with st.expander("🔥 Korelasyon Analizi", expanded=False):
        try:
            st.image("images/correlation_matrix.png", 
                    caption="Değişkenler arası korelasyon matrisi", 
                    use_container_width=True)
            st.info("""
            **En yüksek korelasyonlar:**
            - GenHlth (Genel sağlık) ve Diyabet arasında güçlü ilişki
            - BMI ve HighBP (Yüksek tansiyon) pozitif korelasyon
            - PhysActivity (Fiziksel aktivite) negatif korelasyon
            """)
        except:
            st.warning("Korelasyon matrisi görseli bulunamadı")

# --- MODEL PERFORMANSI SEKMESİ ---
with tab3:
    st.markdown("### 📈 Model Performans Metrikleri")
    st.markdown("LightGBM modelinin performans değerlendirmesi")
    
    # Metrikler
    col_met1, col_met2, col_met3, col_met4 = st.columns(4)
    with col_met1:
        st.metric("Doğruluk", "86%", delta="Yüksek")
    with col_met2:
        st.metric("Precision", "0.84", delta="İyi")
    with col_met3:
        st.metric("Recall", "0.87", delta="Çok İyi")
    with col_met4:
        st.metric("F1 Score", "0.85", delta="Dengeli")
    
    with st.expander("📊 Confusion Matrix", expanded=False):
        try:
            st.image("images/confusion_matrix.png", 
                    caption="Modelin tahmin performansı", 
                    use_container_width=True)
            st.info("""
            **Confusion Matrix Açıklaması:**
            - **True Positive (TP):** Diyabetli olarak doğru tahmin edildi
            - **True Negative (TN):** Diyabetsiz olarak doğru tahmin edildi
            - **False Positive (FP):** Yanlış alarm (Diyabetsiz kişi diyabetli gösterildi)
            - **False Negative (FN):** Kaçırılan vaka (Diyabetli kişi sağlıklı gösterildi)
            """)
        except:
            st.warning("Confusion matrix görseli bulunamadı")
    
    with st.expander("📉 Detaylı Metrikler", expanded=False):
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            try:
                st.image("images/model_metrics_1.png", 
                        caption="ROC Curve ve diğer metrikler",
                        use_container_width=True)
            except:
                st.warning("Metrik görseli 1 bulunamadı")
        
        with col_img2:
            try:
                st.image("images/model_metrics_2.png", 
                        caption="Feature Importance",
                        use_container_width=True)
            except:
                st.warning("Metrik görseli 2 bulunamadı")

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
if st.button("🔬 RİSK ANALİZİNİ BAŞLAT", use_container_width=True):
    # Sadece olasılık oranını alıyoruz (Diyabet olma ihtimali)
    probability = model.predict_proba(input_data)[0][1]
    
    # Progress bar ile görsel efekt
    with st.spinner('Model analiz ediyor...'):
        import time
        time.sleep(0.5)
    
    # --- MongoDB'ye Tahmin Logla ---
    input_features_dict = {
        "HighBP": int(high_bp),
        "HighChol": int(high_chol),
        "BMI": bmi,
        "Smoker": int(smoker),
        "PhysActivity": int(phys_activity),
        "GenHlth": gen_hlth,
        "MentHlth": ment_hlth,
        "PhysHlth": phys_hlth,
        "DiffWalk": int(diff_walk),
        "Sex": sex,
        "Age": age_display,
        "Education": education,
        "Income": income
    }
    
    # Veritabanına kaydet
    try:
        log_result = db.log_prediction(input_features_dict, probability)
        if log_result:
            st.sidebar.success("✅ Tahmin kaydedildi")
        else:
            st.sidebar.warning("⚠️ Kayıt yapılamadı")
    except Exception as e:
        st.sidebar.error(f"DB Hatası: {e}")
    
    # Yüksek risk uyarısı - Alert sistemi
    if probability >= 0.6:
        st.toast("🚨 YÜKSEK RİSKLİ TAHMİN TESPİT EDİLDİ!", icon="🚨")
        st.balloons()
    
    st.markdown("### 📊 Analiz Sonuçları")
    
    # Risk skorunu göster
    col_result1, col_result2 = st.columns([2, 1])
    with col_result1:
        st.metric(label="Hesaplanan Risk Skoru", value=f"%{probability*100:.2f}")
    with col_result2:
        # Risk seviyesi göstergesi
        if probability < 0.3:
            st.success("🟢 Düşük Risk")
        elif probability < 0.6:
            st.warning("🟡 Orta Risk")
        else:
            st.error("🔴 Yüksek Risk")
    
    # Progress bar ile risk seviyesi
    st.progress(min(probability, 1.0))
    
    # --- BUSINESS LOGIC (İŞ MANTIĞI) ---
    # Normalde eşik 0.5'tir. Ancak sağlıkta riski kaçırmamak için
    # eşik değerini 0.3'e çektik. (Recall Optimizasyonu)
    THRESHOLD = 0.3 
    
    st.markdown("---")
    if probability > THRESHOLD:
        st.error(f"### 🚨 DİKKAT: Diyabet Riski Tespit Edildi!")
        st.warning(f"""
        **Model Değerlendirmesi:**
        - Risk Skoru: %{probability*100:.1f}
        - Risk Eşik Değeri: %{THRESHOLD*100}
        
        **Önerilerimiz:**
        - 🏥 En kısa sürede bir sağlık kuruluşuna başvurun
        - 🩸 Açlık kan şekeri testi yaptırın
        - 👨‍⚕️ Bir endokrinoloji uzmanı ile görüşün
        """)
        st.info("⚠️ **Önemli Not:** Bu analiz tıbbi teşhis değildir, sadece risk tahminidir.")
    else:
        st.success(f"### ✅ Sonuç: Diyabet Riski Düşük")
        st.info(f"""
        **Model Değerlendirmesi:**
        - Risk Skoru: %{probability*100:.1f}
        - Risk Eşik Değeri: %{THRESHOLD*100}
        
        **Sağlıklı kalın:**
        - 🥗 Dengeli beslenmeye devam edin
        - 🏃‍♂️ Düzenli egzersiz yapın
        - 🏥 Yıllık kontrol muayenelerinizi aksatmayın
        """)
        
    # Açıklama metni
    st.markdown("---")
    with st.expander("ℹ️ Risk Skoru Nasıl Hesaplanıyor?"):
        st.markdown("""
        **Model Detayları:**
        - **Algoritma:** LightGBM (Gradient Boosting)
        - **Eğitim Verisi:** BRFSS 2015 (253,680 kayıt)
        - **Doğruluk:** ~86%
        - **Risk Eşiği:** %30 (Sağlık güvenliği için optimize edilmiş)
        
        Model, 21 farklı sağlık göstergesini analiz ederek diyabet riski hesaplar.
        Eşik değeri, false negative (hastalığı kaçırma) oranını minimize etmek için
        standart %50'den %30'a düşürülmüştür.
        """)

# --- MONITORING TAB ---
with tab4:
    st.markdown("### 🚨 Alert Monitoring Dashboard")
    st.markdown("Yüksek riskli tahminlerin gerçek zamanlı izlenmesi")
    
    # MongoDB bağlantı durumu
    if not db.is_connected():
        st.error("⚠️ MongoDB bağlantısı kurulamadı!")
        st.info("""
        **MongoDB Kurulumu:**
        1. MongoDB Community Server'ı indirin: https://www.mongodb.com/try/download/community
        2. Kurulum sonrası MongoDB servisinin çalıştığından emin olun
        3. Varsayılan port: 27017
        
        Veya MongoDB Atlas (Cloud) kullanabilirsiniz.
        """)
    else:
        st.sidebar.success("✅ MongoDB bağlantısı aktif")
        
        # Auto-refresh butonu
        col_refresh, col_status = st.columns([1, 3])
        with col_refresh:
            if st.button("🔄 Yenile", use_container_width=True):
                st.rerun()
        with col_status:
            st.caption(f"Son güncelleme: {datetime.now().strftime('%H:%M:%S')}")
        
        st.markdown("---")
        
        # --- ÜST METRİKLER ---
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        total_alerts = db.get_alert_count()
        today_alerts = db.get_today_alert_count()
        week_alerts = db.get_alert_count(days=7)
        total_predictions = db.get_total_predictions()
        
        with col_m1:
            st.metric(
                label="🚨 Bugünkü Alertler",
                value=today_alerts,
                delta=None
            )
        
        with col_m2:
            st.metric(
                label="📅 Bu Hafta",
                value=week_alerts,
                delta=None
            )
        
        with col_m3:
            st.metric(
                label="📊 Toplam Alert",
                value=total_alerts,
                delta=None
            )
        
        with col_m4:
            alert_rate = (total_alerts / total_predictions * 100) if total_predictions > 0 else 0
            st.metric(
                label="📈 Alert Oranı",
                value=f"%{alert_rate:.1f}",
                delta=None
            )
        
        st.markdown("---")
        
        # --- ALERT TREND GRAFİĞİ ---
        st.markdown("#### 📈 Son 7 Günlük Alert Trendi")
        
        trend_data = db.get_alert_trend(days=7)
        
        if trend_data:
            import pandas as pd
            trend_df = pd.DataFrame(trend_data)
            trend_df['date'] = pd.to_datetime(trend_df['date'])
            trend_df = trend_df.set_index('date')
            
            st.line_chart(trend_df['count'], use_container_width=True)
        else:
            st.info("Henüz trend verisi bulunmuyor. Tahmin yapıldıkça veriler burada görünecek.")
        
        st.markdown("---")
        
        # --- RİSK DAĞILIMI ---
        st.markdown("#### 🎯 Risk Dağılımı")
        
        risk_dist = db.get_risk_distribution()
        
        col_risk1, col_risk2, col_risk3 = st.columns(3)
        
        with col_risk1:
            st.markdown("""
            <div style="background-color: #0d5016; padding: 20px; border-radius: 10px; text-align: center;">
                <h2 style="color: #4ade80; margin: 0;">🟢 {}</h2>
                <p style="color: #86efac; margin: 5px 0 0 0;">Düşük Risk</p>
            </div>
            """.format(risk_dist['low']), unsafe_allow_html=True)
        
        with col_risk2:
            st.markdown("""
            <div style="background-color: #713f12; padding: 20px; border-radius: 10px; text-align: center;">
                <h2 style="color: #facc15; margin: 0;">🟡 {}</h2>
                <p style="color: #fde047; margin: 5px 0 0 0;">Orta Risk</p>
            </div>
            """.format(risk_dist['medium']), unsafe_allow_html=True)
        
        with col_risk3:
            st.markdown("""
            <div style="background-color: #7f1d1d; padding: 20px; border-radius: 10px; text-align: center;">
                <h2 style="color: #f87171; margin: 0;">🔴 {}</h2>
                <p style="color: #fca5a5; margin: 5px 0 0 0;">Yüksek Risk</p>
            </div>
            """.format(risk_dist['high']), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # --- SON YÜKSEK RİSKLİ TAHMİNLER ---
        st.markdown("#### 🚨 Son Yüksek Riskli Tahminler (Alertler)")
        
        alerts = db.get_alerts(limit=20)
        
        if alerts:
            # Alert listesini tablo olarak göster
            alert_table_data = []
            for alert in alerts:
                features = alert.get('input_features', {})
                alert_table_data.append({
                    "Zaman": alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                    "Risk Skoru": f"%{alert['risk_score']*100:.1f}",
                    "Yaş Grubu": features.get('Age', '-'),
                    "BMI": features.get('BMI', '-'),
                    "Yüksek Tansiyon": "Evet" if features.get('HighBP') else "Hayır",
                    "Yüksek Kolesterol": "Evet" if features.get('HighChol') else "Hayır",
                    "Sigara": "Evet" if features.get('Smoker') else "Hayır"
                })
            
            alert_df = pd.DataFrame(alert_table_data)
            
            # Tabloyu göster
            st.dataframe(
                alert_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Zaman": st.column_config.TextColumn("⏰ Zaman"),
                    "Risk Skoru": st.column_config.TextColumn("📊 Risk Skoru"),
                    "Yaş Grubu": st.column_config.NumberColumn("👤 Yaş Grubu"),
                    "BMI": st.column_config.NumberColumn("⚖️ BMI", format="%.1f"),
                    "Yüksek Tansiyon": st.column_config.TextColumn("💓 Yük. Tansiyon"),
                    "Yüksek Kolesterol": st.column_config.TextColumn("🩸 Yük. Kolesterol"),
                    "Sigara": st.column_config.TextColumn("🚬 Sigara")
                }
            )
            
            # Son alert için detaylı bilgi
            with st.expander("📋 Son Alert Detayları", expanded=False):
                if alerts:
                    last_alert = alerts[0]
                    st.json(last_alert.get('input_features', {}))
        else:
            st.info("🎉 Henüz yüksek riskli tahmin (alert) bulunmuyor.")
            st.caption("Risk skoru %60 ve üzeri olan tahminler burada listelenecektir.")
        
        # --- İSTATİSTİKLER ---
        st.markdown("---")
        with st.expander("📊 Detaylı İstatistikler", expanded=False):
            col_stat1, col_stat2 = st.columns(2)
            
            with col_stat1:
                avg_score = db.get_average_risk_score()
                st.metric("Ortalama Risk Skoru", f"%{avg_score*100:.1f}")
                
                avg_score_week = db.get_average_risk_score(days=7)
                st.metric("Son 7 Gün Ortalama", f"%{avg_score_week*100:.1f}")
            
            with col_stat2:
                st.metric("Toplam Tahmin Sayısı", total_predictions)
                
                if total_predictions > 0:
                    high_risk_rate = (risk_dist['high'] / total_predictions) * 100
                    st.metric("Yüksek Risk Oranı", f"%{high_risk_rate:.1f}")