import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Prediksi Risiko Stunting",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    
    .info-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #fff3cd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #d4edda;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: bold;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="main-header">
    <h1>🏥 KLasifikasi Keluarga Rentan Stunting</h1>
    <p>Model Stacked LSTM untuk Deteksi Dini Risiko Stunting pada keluarga rentan stunting</p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 📊 Menu Navigasi")
    page = st.selectbox("Pilih Halaman:", ["🔍 Klasifikasi", "📈 Informasi Model", "❓ Bantuan"])
    
    st.markdown("---")
    st.markdown("### 🎯 Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini menggunakan teknologi **stacked LSTM** 
    untuk mengklasifikasi keluaraga rentan stunting berdasarkan kondisi 
    keluarga dan lingkungan.
    """)

if page == "🔍 Klasifikasi":
    # --- Load Model dan Scaler ---
    try:
        model = load_model("model_lstm_stunting.h5")
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        model_loaded = True
    except:
        st.error("⚠️ Model atau scaler tidak ditemukan! Pastikan file model_lstm_stunting.h5 dan scaler.pkl tersedia.")
        model_loaded = False
    
    if model_loaded:
        # --- Form Input ---
        st.markdown("### 📝 Input Data Keluarga")
        
        with st.expander("ℹ️ Petunjuk Pengisian", expanded=False):
            st.markdown("""
            **Silakan isi semua kondisi keluarga di bawah ini:**
            - **Baduta**: Bayi Di Bawah Dua Tahun (0-24 bulan)
            - **Balita**: Bayi Di Bawah Lima Tahun (0-59 bulan)
            - **PUS**: Pasangan Usia Subur
            - **PUS Hamil**: Pasangan Usia Subur yang sedang hamil
            - **4T**: Terlalu muda, tua, dekat, banyak (faktor risiko kehamilan)
            """)
        
        # Membuat form yang lebih terstruktur
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 👶 Kondisi Anak")
                baduta = st.radio("Memiliki Baduta (0-24 bulan)?", 
                                ["❌ Tidak", "✅ Ya"], key="baduta")
                balita = st.radio("Memiliki Balita (0-59 bulan)?", 
                                ["❌ Tidak", "✅ Ya"], key="balita")
                
                st.markdown("#### 🏠 Kondisi Sanitasi")
                sumber_air = st.radio("Sumber Air Tidak Layak?", 
                                    ["❌ Tidak", "✅ Ya"], key="air")
                jamban = st.radio("Jamban/WC Tidak Layak?", 
                                ["❌ Tidak", "✅ Ya"], key="jamban")
            
            with col2:
                st.markdown("#### 👩‍👧‍👦 Kondisi Ibu")
                pus = st.radio("Termasuk PUS (Pasangan Usia Subur)?", 
                             ["❌ Tidak", "✅ Ya"], key="pus")
                pus_hamil = st.radio("PUS sedang Hamil?", 
                                   ["❌ Tidak", "✅ Ya"], key="hamil")
                bukan_peserta_kb = st.radio("Bukan Peserta KB Modern?", 
                                          ["❌ Tidak", "✅ Ya"], key="kb")
            
            with col3:
                st.markdown("#### ⚠️ Faktor Risiko 4T")
                terlalu_muda = st.radio("Terlalu Muda (< 20 tahun)?", 
                                      ["❌ Tidak", "✅ Ya"], key="muda")
                terlalu_tua = st.radio("Terlalu Tua (> 35 tahun)?", 
                                     ["❌ Tidak", "✅ Ya"], key="tua")
                terlalu_dekat = st.radio("Jarak Kelahiran Terlalu Dekat (< 2 tahun)?", 
                                       ["❌ Tidak", "✅ Ya"], key="dekat")
                terlalu_banyak = st.radio("Anak Terlalu Banyak (> 4 anak)?", 
                                        ["❌ Tidak", "✅ Ya"], key="banyak")
            
            # Submit button
            submitted = st.form_submit_button("🔍 Analisis Risiko Stunting", use_container_width=True)
        
        if submitted:
            # --- Konversi ke bentuk numerik ---
            input_data = {
                "baduta": 1 if baduta == "✅ Ya" else 0,
                "balita": 1 if balita == "✅ Ya" else 0,
                "pus": 1 if pus == "✅ Ya" else 0,
                "pus_hamil": 1 if pus_hamil == "✅ Ya" else 0,
                "sumber_air_layak_tidak": 1 if sumber_air == "✅ Ya" else 0,
                "jamban_layak_tidak": 1 if jamban == "✅ Ya" else 0,
                "terlalu_muda": 1 if terlalu_muda == "✅ Ya" else 0,
                "terlalu_tua": 1 if terlalu_tua == "✅ Ya" else 0,
                "terlalu_dekat": 1 if terlalu_dekat == "✅ Ya" else 0,
                "terlalu_banyak": 1 if terlalu_banyak == "✅ Ya" else 0,
                "bukan_peserta_kb_modern": 1 if bukan_peserta_kb == "✅ Ya" else 0,
            }
            
            # Preprocessing
            input_df = pd.DataFrame([input_data])
            scaled_input = scaler.transform(input_df)
            reshaped_input = scaled_input.reshape((1, 1, input_df.shape[1]))
            
            # --- Prediksi ---
            with st.spinner("🔄 Sedang menganalisis data..."):
                pred = model.predict(reshaped_input)[0][0]
                risk_percentage = pred * 100
                
            # --- Hasil Prediksi ---
            st.markdown("---")
            st.markdown("## 📊 Hasil Analisis")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                # Gauge Chart untuk risk score
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = risk_percentage,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Tingkat Keluarga Rentan Stunting (%)"},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkred" if pred > 0.7 else "orange" if pred > 0.3 else "darkgreen"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # Status dan rekomendasi
            if pred > 0.7:
                st.markdown("""
                <div class="warning-box">
                    <h3>⚠️ RISIKO TINGGI</h3>
                    <p><strong>Tingkat Risiko:</strong> {:.1f}%</p>
                    <p>Keluarga ini memiliki risiko tinggi stunting. Diperlukan intervensi segera!</p>
                </div>
                """.format(risk_percentage), unsafe_allow_html=True)
                
                st.markdown("#### 🎯 Rekomendasi Tindakan:")
                recommendations = [
                    "Segera konsultasi dengan tenaga kesehatan",
                    "Perbaiki kualitas sanitasi (air bersih dan jamban layak)",
                    "Ikuti program KB modern untuk mengatur jarak kelahiran",
                    "Tingkatkan asupan gizi ibu dan anak",
                    "Rutin periksa tumbuh kembang anak di posyandu"
                ]
                
            elif pred > 0.3:
                st.markdown("""
                <div class="warning-box">
                    <h3>⚠️ RISIKO SEDANG</h3>
                    <p><strong>Tingkat Risiko:</strong> {:.1f}%</p>
                    <p>Keluarga ini memiliki risiko sedang stunting. Perlu perhatian khusus.</p>
                </div>
                """.format(risk_percentage), unsafe_allow_html=True)
                
                st.markdown("#### 🎯 Rekomendasi Tindakan:")
                recommendations = [
                    "Konsultasi rutin dengan bidan atau dokter",
                    "Perhatikan kualitas air minum dan sanitasi",
                    "Pertimbangkan program KB untuk mengatur kelahiran",
                    "Perbaiki pola makan keluarga",
                    "Aktif mengikuti kegiatan posyandu"
                ]
                
            else:
                st.markdown("""
                <div class="success-box">
                    <h3>✅ RISIKO RENDAH</h3>
                    <p><strong>Tingkat Risiko:</strong> {:.1f}%</p>
                    <p>Keluarga ini memiliki risiko rendah stunting. Pertahankan kondisi yang baik!</p>
                </div>
                """.format(risk_percentage), unsafe_allow_html=True)
                
                st.markdown("#### 🎯 Rekomendasi Pemeliharaan:")
                recommendations = [
                    "Pertahankan pola hidup sehat",
                    "Terus pantau tumbuh kembang anak",
                    "Jaga kualitas sanitasi yang sudah baik",
                    "Rutin kontrol kesehatan keluarga",
                    "Terus ikuti program posyandu"
                ]
            
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"{i}. {rec}")
            
            # Risk factors analysis
            st.markdown("#### 📈 Analisis Faktor Risiko")
            risk_factors = []
            if input_data["baduta"]: risk_factors.append("Memiliki Baduta")
            if input_data["balita"]: risk_factors.append("Memiliki Balita") 
            if input_data["sumber_air_layak_tidak"]: risk_factors.append("Air Tidak Layak")
            if input_data["jamban_layak_tidak"]: risk_factors.append("Jamban Tidak Layak")
            if input_data["terlalu_muda"]: risk_factors.append("Ibu Terlalu Muda")
            if input_data["terlalu_tua"]: risk_factors.append("Ibu Terlalu Tua")
            if input_data["terlalu_dekat"]: risk_factors.append("Jarak Kelahiran Terlalu Dekat")
            if input_data["terlalu_banyak"]: risk_factors.append("Anak Terlalu Banyak")
            if input_data["bukan_peserta_kb_modern"]: risk_factors.append("Bukan Peserta KB Modern")
            
            if risk_factors:
                st.warning(f"**Faktor risiko yang teridentifikasi:** {', '.join(risk_factors)}")
            else:
                st.success("**Tidak ada faktor risiko utama yang teridentifikasi.**")

elif page == "📈 Informasi Model":
    st.markdown("### 🤖 Informasi Model LSTM")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>📊 Spesifikasi Model</h4>
            <ul>
                <li><strong>Algoritma:</strong> LSTM (Long Short-Term Memory)</li>
                <li><strong>Input Features:</strong> 11 variabel</li>
                <li><strong>Output:</strong> Probabilitas risiko stunting</li>
                <li><strong>Threshold:</strong> 0.5 (50%)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h4>🎯 Variabel Input</h4>
            <ul>
                <li>Status Baduta & Balita</li>
                <li>Kondisi PUS & Kehamilan</li>
                <li>Kualitas Air & Sanitasi</li>
                <li>Faktor Risiko 4T</li>
                <li>Status Kepesertaan KB</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

elif page == "❓ Bantuan":
    st.markdown("### 🆘 Bantuan & FAQ")
    
    with st.expander("❓ Apa itu Stunting?"):
        st.markdown("""
        Stunting adalah kondisi gagal tumbuh pada anak balita akibat kekurangan gizi kronis, 
        terutama dalam 1000 hari pertama kehidupan (HPK). Anak yang mengalami stunting akan 
        memiliki tinggi badan yang lebih pendek dari standar usianya.
        """)
    
    with st.expander("❓ Bagaimana cara menggunakan aplikasi ini?"):
        st.markdown("""
        1. Pilih halaman "🔍 Prediksi" di sidebar
        2. Isi semua kondisi keluarga sesuai dengan keadaan sebenarnya
        3. Klik tombol "🔍 Analisis Risiko Stunting"
        4. Lihat hasil prediksi dan ikuti rekomendasi yang diberikan
        """)
    
    with st.expander("❓ Apa arti dari faktor risiko 4T?"):
        st.markdown("""
        **4T** adalah faktor risiko dalam kehamilan:
        - **Terlalu Muda**: Ibu hamil berusia < 20 tahun
        - **Terlalu Tua**: Ibu hamil berusia > 35 tahun  
        - **Terlalu Dekat**: Jarak kehamilan < 2 tahun
        - **Terlalu Banyak**: Memiliki anak > 4 orang
        """)
    
    with st.expander("❓ Seberapa akurat prediksi ini?"):
        st.markdown("""
        Model ini menggunakan teknologi Deep Learning stacked LSTM yang telah dilatih dengan data historis.
        Namun, hasil prediksi ini hanya sebagai **skrining awal** dan tidak menggantikan 
        diagnosis medis profesional. Selalu konsultasikan dengan tenaga kesehatan untuk 
        penanganan yang tepat.
        """)

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>💡 <strong>Disclaimer:</strong> Aplikasi ini hanya untuk skrining awal. 
    Konsultasikan dengan tenaga kesehatan untuk diagnosis dan penanganan yang tepat.</p>
    <p>🏥 Dikembangkan untuk mendukung program pencegahan keluarga rentan stunting</p>
</div>
""", unsafe_allow_html=True)