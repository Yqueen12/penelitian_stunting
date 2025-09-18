import streamlit as st
import numpy as np
import pandas as pd
import pickle 
from tensorflow.keras.models import load_model

# Konfigurasi halaman
st.set_page_config(
    page_title="Klasifikasi Keluarga Rentan Stunting",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        padding: 2.5rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    .risk-box {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
    }
    .high-risk {
        background: #dc3545;
        color: white;
    }
    .low-risk {
        background: #28a745;
        color: white;
    }
    .cream-box {
        background: #e0e0e0;
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 1rem;
    }
    .factor-list {
        margin-top: 0.5rem;
        padding-left: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header aplikasi
st.markdown("""
<div class="main-header">
    <h1>Klasifikasi Keluarga Rentan Stunting</h1>
</div>
""", unsafe_allow_html=True)

# Fungsi memuat model dan scaler
def load_ml_components():
    try:
        model = load_model("model_lstm_stunting.h5")
        with open("scaler.pkl", "rb") as file:
            scaler = pickle.load(file)
        return model, scaler, True
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        return None, None, False

# Fungsi analisis faktor risiko
def analyze_risk_factors(input_data):
    factor_mapping = {
        "baduta": "Ada anak usia 0-24 bulan",
        "balita": "Ada anak usia 0-59 bulan",
        "sumber_air_layak_tidak": "Air minum tidak layak",
        "jamban_layak_tidak": "Jamban tidak layak",
        "terlalu_muda": "Ibu hamil di usia < 20 tahun",
        "terlalu_tua": "Ibu hamil di usia > 35 tahun",
        "terlalu_dekat": "Jarak kelahiran < 2 tahun",
        "terlalu_banyak": "Jumlah anak lebih dari 4",
        "bukan_peserta_kb_modern": "Tidak menggunakan KB modern"
    }
    identified_factors = []
    for key, value in input_data.items():
        if value == 1 and key in factor_mapping:
            identified_factors.append(factor_mapping[key])
    return identified_factors

# Load model
model, scaler, model_status = load_ml_components()

if model_status:
    st.markdown("### Input Data Kondisi Keluarga")

    with st.form("family_risk_assessment"):
        col1, col2, col3 = st.columns(3)
        with col1:
            has_baduta = st.radio("Memiliki anak Baduta (0-24 bulan)", ["Tidak", "Ya"])
            has_balita = st.radio("Memiliki anak Balita (0-59 bulan)", ["Tidak", "Ya"])
            water_quality = st.radio("Sumber air tidak layak konsumsi", ["Tidak", "Ya"])
            sanitation_quality = st.radio("Jamban tidak memenuhi standar", ["Tidak", "Ya"])
        with col2:
            pus_status = st.radio("Termasuk Pasangan Usia Subur (PUS)", ["Tidak", "Ya"])
            pregnancy_status = st.radio("Sedang hamil", ["Tidak", "Ya"])
            kb_participation = st.radio("Tidak menggunakan KB modern", ["Tidak", "Ya"])
        with col3:
            age_young = st.radio("Ibu hamil terlalu muda (< 20 th)", ["Tidak", "Ya"])
            age_old = st.radio("Ibu hamil terlalu tua (> 35 th)", ["Tidak", "Ya"])
            birth_spacing = st.radio("Jarak kelahiran < 2 tahun", ["Tidak", "Ya"])
            children_count = st.radio("Jumlah anak > 4", ["Tidak", "Ya"])
        
        submit_analysis = st.form_submit_button("Analisis Risiko", use_container_width=True)

    if submit_analysis:
        family_data = {
            "baduta": 1 if has_baduta == "Ya" else 0,
            "balita": 1 if has_balita == "Ya" else 0,
            "pus": 1 if pus_status == "Ya" else 0,
            "pus_hamil": 1 if pregnancy_status == "Ya" else 0,
            "sumber_air_layak_tidak": 1 if water_quality == "Ya" else 0,
            "jamban_layak_tidak": 1 if sanitation_quality == "Ya" else 0,
            "terlalu_muda": 1 if age_young == "Ya" else 0,
            "terlalu_tua": 1 if age_old == "Ya" else 0,
            "terlalu_dekat": 1 if birth_spacing == "Ya" else 0,
            "terlalu_banyak": 1 if children_count == "Ya" else 0,
            "bukan_peserta_kb_modern": 1 if kb_participation == "Ya" else 0,
        }
        
        input_dataframe = pd.DataFrame([family_data])
        scaled_data = scaler.transform(input_dataframe)
        lstm_input = scaled_data.reshape((1, 1, input_dataframe.shape[1]))
        
        with st.spinner("Sedang menganalisis..."):
            prediction_result = model.predict(lstm_input)[0][0]
        
        st.markdown("---")
        st.markdown("## Hasil Analisis")

        identified_risks = analyze_risk_factors(family_data)
        
        # Kotak hasil utama - tanpa persentase
        if prediction_result >= 0.5:
            st.markdown(f"""
            <div class="risk-box high-risk">
                <h3>Berisiko</h3>
                <p>Keluarga teridentifikasi berisiko stunting</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-box low-risk">
                <h3>Tidak Berisiko</h3>
                <p>Keluarga teridentifikasi tidak berisiko stunting</p>
            </div>
            """, unsafe_allow_html=True)

        # Kotak faktor risiko
        st.markdown(f"""
        <div class="cream-box">
            <h3>Faktor Risiko</h3>
            <div class="factor-list">
                {"<br>".join(identified_risks) if identified_risks else "Tidak ada faktor risiko utama yang terdeteksi."}
            </div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; background: #f8f9fa; padding: 1.5rem; border-radius: 8px; margin-top: 2rem;">
    <p><strong>Penerapan Algoritma Stacked LSTM Untuk Klasifikasi dan Visualisasi Keluarga Rentan Stunting.</strong></p>
</div>
""", unsafe_allow_html=True)