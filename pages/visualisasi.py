import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import base64
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ========== Konfigurasi Awal ========== #
st.set_page_config(page_title="Peta Risiko Stunting", layout="wide", initial_sidebar_state="expanded")

# Custom CSS untuk tampilan lebih menarik
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
        
        .main > div {
            font-family: 'Poppins', sans-serif;
        }
        
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin: 10px 0;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        }
        
        .metric-number {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 5px;
        }
        
        .metric-label {
            font-size: 1rem;
            opacity: 0.9;
        }
        
        .section-header {
            color: #667eea;
            font-size: 1.8rem;
            font-weight: 600;
            margin: 30px 0 20px 0;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        
        .info-box {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 15px;
            border-radius: 10px;
            color: white;
            margin: 10px 0;
        }
        
        .sidebar .stSelectbox > label {
            font-weight: 600;
            color: #667eea;
        }
        
        .legend-container {
            border: 1px solid #667eea;
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
        }
        
        .content-container {
            max-width: 800px;
            margin: 0;
        }
        
        .chart-container {
            margin: 20px 0;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
        }
        
        .table-container {
            margin: 20px 0;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
        }
    </style>
""", unsafe_allow_html=True)

# Fungsi: Konversi gambar PNG ke base64
def load_icon_base64(path):
    try:
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except FileNotFoundError:
        return None

# Load ikon gambar sebagai base64
icon_red = load_icon_base64('assets/marker_red.png')
icon_green = load_icon_base64('assets/marker_green.png')

# Fungsi: Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("penelitian_bersih.xlsx")
        df.columns = df.columns.str.lower()

        # Normalisasi data risiko
        df['risiko_stunting'] = df['risiko_stunting'].astype(str).str.strip().str.title()
        df['risiko_stunting'] = df['risiko_stunting'].replace({
            '1': 'Berisiko', '0': 'Tidak Berisiko',
            'True': 'Berisiko', 'False': 'Tidak Berisiko',
            'Yes': 'Berisiko', 'No': 'Tidak Berisiko'
        })
        return df
    except FileNotFoundError:
        st.error("File 'penelitian_bersih.xlsx' tidak ditemukan!")
        return pd.DataFrame()

# Fungsi: Generate Map per Kelurahan (1 marker per kelurahan)
def generate_map(df):
    if df.empty:
        return None
        
    df = df.dropna(subset=['lat', 'lon'])
    
    if df.empty:
        return None

    # Ambil satu titik representatif per kelurahan
    kelurahan_summary = df.groupby('namakelurahan').agg({
        'lat': 'mean',
        'lon': 'mean',
        'risiko_stunting': lambda x: x.mode()[0] if len(x) > 0 else 'Tidak Berisiko',
        'namakecamatan': 'first'
    }).reset_index()

    distribusi = df.groupby(['namakelurahan', 'risiko_stunting']).size().unstack(fill_value=0).reset_index()
    map_data = pd.merge(kelurahan_summary, distribusi, on='namakelurahan', how='left')

    m = folium.Map(location=[df['lat'].mean(), df['lon'].mean()], zoom_start=12)

    for _, row in map_data.iterrows():
        # Gunakan ikon default jika custom icon tidak tersedia
        if icon_red and icon_green and row['risiko_stunting'].lower() == 'berisiko':
            icon_url = f"data:image/png;base64,{icon_red}"
            icon = folium.CustomIcon(icon_url, icon_size=(30, 30))
        elif icon_red and icon_green:
            icon_url = f"data:image/png;base64,{icon_green}"
            icon = folium.CustomIcon(icon_url, icon_size=(30, 30))
        else:
            # Fallback ke ikon default
            color = 'red' if row['risiko_stunting'].lower() == 'berisiko' else 'green'
            icon = folium.Icon(color=color, icon='info-sign')

        popup_html = f"""
        <div style="font-size: 14px; font-family: 'Poppins', sans-serif;">
            <b style="color: #667eea;">📍 Kelurahan:</b> {row['namakelurahan']}<br>
            <b style="color: #667eea;">🏘️ Kecamatan:</b> {row['namakecamatan']}<br>
            <b style="color: #f5576c;">📊 Status Dominan:</b> {row['risiko_stunting']}<br><br>
            <b style="color: #667eea;">📈 Distribusi Data:</b><br>
            ✅ Tidak Berisiko: <b>{row.get('Tidak Berisiko', 0)}</b><br>
            ⚠️ Berisiko: <b>{row.get('Berisiko', 0)}</b><br>
            <b style="color: #764ba2;">📊 Total: {row.get('Tidak Berisiko', 0) + row.get('Berisiko', 0)}</b>
        </div>
        """

        folium.Marker(
            location=[row['lat'], row['lon']],
            icon=icon,
            popup=folium.Popup(popup_html, max_width=350)
        ).add_to(m)

    return m

# Fungsi: Membuat visualisasi distribusi (tanpa tren waktu)
def create_distribution_charts(df):
    if df.empty:
        return None, None, None
    
    # 1. Pie Chart - Distribusi Keseluruhan
    risk_counts = df['risiko_stunting'].value_counts()
    
    fig_pie = px.pie(
        values=risk_counts.values, 
        names=risk_counts.index,
        title="Distribusi Risiko Stunting",
        color_discrete_map={
            'Berisiko': '#ff6b6b',
            'Tidak Berisiko': '#51cf66'
        }
    )
    fig_pie.update_layout(
        font_family="Poppins",
        title_font_size=14,
        title_x=0.5,
        height=350,
        margin=dict(t=40, b=10, l=10, r=10)
    )
    
    # 2. Bar Chart - Distribusi per Kecamatan
    kec_dist = df.groupby(['namakecamatan', 'risiko_stunting']).size().unstack(fill_value=0)
    
    fig_bar_kec = px.bar(
        kec_dist, 
        title="Distribusi per Kecamatan",
        color_discrete_map={
            'Berisiko': '#ff6b6b',
            'Tidak Berisiko': '#51cf66'
        },
        barmode='group'
    )
    fig_bar_kec.update_layout(
        font_family="Poppins",
        title_font_size=14,
        title_x=0.5,
        xaxis_title="Kecamatan",
        yaxis_title="Jumlah Kasus",
        height=350,
        margin=dict(t=40, b=40, l=40, r=10)
    )
    
    # 3. Bar Chart - Distribusi per Kelurahan (Top 10)
    kel_dist = df.groupby(['namakelurahan', 'risiko_stunting']).size().unstack(fill_value=0)
    kel_total = kel_dist.sum(axis=1).sort_values(ascending=False).head(10)
    kel_dist_top = kel_dist.loc[kel_total.index]
    
    fig_bar_kel = px.bar(
        kel_dist_top, 
        title="Top 10 Kelurahan dengan Kasus Terbanyak",
        color_discrete_map={
            'Berisiko': '#ff6b6b',
            'Tidak Berisiko': '#51cf66'
        },
        barmode='stack'
    )
    fig_bar_kel.update_layout(
        font_family="Poppins",
        title_font_size=14,
        title_x=0.5,
        xaxis_title="Kelurahan",
        yaxis_title="Jumlah Kasus",
        height=400,
        margin=dict(t=40, b=60, l=40, r=10)
    )
    
    return fig_pie, fig_bar_kec, fig_bar_kel

# ========== Main App ========== #
def main():
    # Header dengan gradient
    st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="background: linear-gradient(90deg, #667eea, #764ba2); 
                       background-clip: text; -webkit-background-clip: text; 
                       -webkit-text-fill-color: transparent; 
                       font-size: 3rem; font-weight: 700; margin: 0;">
                🗺️ Peta Risiko Stunting Kota Bogor
            </h1>
        </div>
    """, unsafe_allow_html=True)

    # Load data
    df = load_data()
    
    if df.empty:
        st.error("Tidak dapat memuat data. Pastikan file 'penelitian_bersih.xlsx' tersedia.")
        return

    # Sidebar Filter dengan styling
    with st.sidebar:
        st.markdown("""
            <div style="text-align: center; padding: 20px 0;">
                <h2 style="color: #667eea; font-weight: 700;">🔍 Filter Data</h2>
            </div>
        """, unsafe_allow_html=True)
        
        kec = ['Semua'] + sorted(df['namakecamatan'].unique())
        kel = ['Semua'] + sorted(df['namakelurahan'].unique())
        
        kecamatan = st.selectbox("📍 Pilih Kecamatan", kec)
        kelurahan = st.selectbox("🏘️ Pilih Kelurahan", kel)
        
        if 'tahun' in df.columns:
            tahun = ['Semua'] + sorted(df['tahun'].unique(), reverse=True)
            tahun_select = st.selectbox("📅 Pilih Tahun", tahun)
        else:
            tahun_select = 'Semua'

        # Info box di sidebar
        st.markdown("""
            <div class="info-box">
                <h4>ℹ️ Informasi</h4>
                <p>• Marker Merah: Kelurahan Berisiko<br>
                • Marker Hijau: Kelurahan Tidak Berisiko<br>
                • Klik marker untuk detail</p>
            </div>
        """, unsafe_allow_html=True)

    # Filter DataFrame
    df_filtered = df.copy()
    if kecamatan != 'Semua':
        df_filtered = df_filtered[df_filtered['namakecamatan'] == kecamatan]
    if kelurahan != 'Semua':
        df_filtered = df_filtered[df_filtered['namakelurahan'] == kelurahan]
    if tahun_select != 'Semua' and 'tahun' in df.columns:
        df_filtered = df_filtered[df_filtered['tahun'] == tahun_select]

    if df_filtered.empty:
        st.warning("❗ Tidak ada data untuk filter yang dipilih.")
        return

    # Metrics Cards
    col1, col2, col3, col4 = st.columns(4)
    
    total_data = len(df_filtered)
    berisiko = len(df_filtered[df_filtered['risiko_stunting'] == 'Berisiko'])
    tidak_berisiko = len(df_filtered[df_filtered['risiko_stunting'] == 'Tidak Berisiko'])
    persentase_risiko = (berisiko / total_data * 100) if total_data > 0 else 0
    
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-number">{total_data:,}</div>
                <div class="metric-label">Total Data</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);">
                <div class="metric-number">{berisiko:,}</div>
                <div class="metric-label">Berisiko</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);">
                <div class="metric-number">{tidak_berisiko:,}</div>
                <div class="metric-label">Tidak Berisiko</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #ffd43b 0%, #fab005 100%);">
                <div class="metric-number">{persentase_risiko:.1f}%</div>
                <div class="metric-label">% Berisiko</div>
            </div>
        """, unsafe_allow_html=True)

    # Peta
    st.markdown('<h2 class="section-header">🗺️ Peta Interaktif</h2>', unsafe_allow_html=True)
    
    # Legend
    st.markdown("""
        <div class="legend-container">
            <h4 style="margin-top: 0; color: #667eea;">📍 Legenda Peta</h4>
            <div style="display: flex; justify-content: space-around;">
                <div style="display: flex; align-items: center;">
                    <div style="width: 20px; height: 20px; background-color: #ff6b6b; border-radius: 50%; margin-right: 10px;"></div>
                    <span>Kelurahan Berisiko</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 20px; height: 20px; background-color: #51cf66; border-radius: 50%; margin-right: 10px;"></div>
                    <span>Kelurahan Tidak Berisiko</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    map_obj = generate_map(df_filtered)
    if map_obj:
        st_folium(map_obj, height=500, width=None, key="peta_stunting")
    else:
        st.error("Tidak dapat menampilkan peta. Pastikan data koordinat tersedia.")

    # Visualisasi Distribusi
    st.markdown('<h2 class="section-header">📊 Analisis Data</h2>', unsafe_allow_html=True)
    
    fig_pie, fig_bar_kec, fig_bar_kel = create_distribution_charts(df_filtered)
    
    if fig_pie:
        # Row 1: Pie chart dan bar chart kecamatan
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            if fig_bar_kec is not None:
                st.plotly_chart(fig_bar_kec, use_container_width=True)
        
        # Row 2: Bar chart kelurahan (full width)
        if fig_bar_kel is not None:
            st.plotly_chart(fig_bar_kel, use_container_width=True)
    
    # Tabel Detail
    st.markdown('<h2 class="section-header">📋 Tabel Detail Data</h2>', unsafe_allow_html=True)
    
    # Summary table
    summary_df = df_filtered.groupby(['namakecamatan', 'namakelurahan', 'risiko_stunting']).size().unstack(fill_value=0).reset_index()
    summary_df['Total'] = summary_df.get('Berisiko', 0) + summary_df.get('Tidak Berisiko', 0)
    summary_df = summary_df.sort_values('Total', ascending=False)
    
    st.dataframe(
        summary_df,
        use_container_width=True,
        height=300
    )
    
    # Download button
    csv = summary_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Data sebagai CSV",
        data=csv,
        file_name=f"data_stunting_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

    # Footer
    st.markdown("""
        <div style="text-align: center; padding: 30px 0 10px 0; color: #666;">
            <hr style="border: 1px solid #eee;">
            <p>Peta Keluarga Risiko Stunting  - Kota Bogor </p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()