import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from google import genai
import plotly.express as px
import plotly.graph_objects as go
import json
import pydeck as pdk
import io
from sklearn.linear_model import LinearRegression

# ==========================================
# KONFIGURASI HALAMAN & SESSION STATE
# ==========================================
st.set_page_config(page_title="Dashboard EBT Kulon Progo - Oji", layout="wide")

if "ai_result_text" not in st.session_state: st.session_state.ai_result_text = None
if "ai_result_model" not in st.session_state: st.session_state.ai_result_model = None
if "ai_chart_data" not in st.session_state: st.session_state.ai_chart_data = None

st.title("⚡ Dashboard Analisis Potensi EBT & MCDM Kabupaten Kulon Progo")
st.markdown("**Oleh: Muhammad Fachrurrozy (Teknik Fisika UGM)**")
st.divider()

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.header("📂 1. Input Data Sistem")
st.sidebar.caption("Support File: Excel manual (.xlsx) atau Raw NASA POWER (.csv)")
uploaded_file = st.sidebar.file_uploader("Upload File Data", type=["xlsx", "xls", "csv", "json"])
st.sidebar.divider()

st.sidebar.header("🌍 2. Asumsi Makro & Kebijakan")
inflasi = st.sidebar.number_input("Tingkat Inflasi Tahunan (%)", 0.0, 15.0, 3.5, 0.1) / 100
kebijakan = st.sidebar.selectbox("Skenario Kebijakan", ["Business as Usual (BAU)", "Pajak Karbon Tinggi (Pro-Lingkungan)", "Subsidi Masif EBT (Pro-Ekonomi)"])
tahun_evaluasi = st.sidebar.slider("Tahun Target Evaluasi", 2025, 2060, 2060)
selisih_tahun = tahun_evaluasi - 2025
st.sidebar.divider()

st.sidebar.header("🧠 3. Integrasi AI")
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    st.sidebar.success("✅ AI terhubung via Server Key.")
except Exception:
    api_key = st.sidebar.text_input("Gemini API Key", type="password")

# ==========================================
# DATA LOKASI EBT (Untuk Peta)
# ==========================================
df_lokasi = pd.DataFrame([
    {"Teknologi": "PLTMH (Air)", "Kecamatan": "Girimulyo", "Lat": -7.7470, "Lon": 110.1260, "Elevasi": 350, "Potensi_MW": 8.5, "Investasi_M_IDR": 18.5, "Color": "blue", "Icon": "tint", "APBN_Tahap": "RPJMN 2025-2029", "Waktu_Optimal": "2026-2027"},
    {"Teknologi": "PLTMH (Air)", "Kecamatan": "Samigaluh", "Lat": -7.6710, "Lon": 110.1700, "Elevasi": 420, "Potensi_MW": 6.2, "Investasi_M_IDR": 14.0, "Color": "blue", "Icon": "tint", "APBN_Tahap": "RPJMN 2025-2029", "Waktu_Optimal": "2027-2028"},
    {"Teknologi": "PLTS (Surya)", "Kecamatan": "Wates", "Lat": -7.8600, "Lon": 110.1400, "Elevasi": 15, "Potensi_MW": 22.0, "Investasi_M_IDR": 28.0, "Color": "orange", "Icon": "sun", "APBN_Tahap": "RPJMN 2025-2029", "Waktu_Optimal": "2025-2026"},
    {"Teknologi": "PLTB (Angin)", "Kecamatan": "Pantai Glagah", "Lat": -7.9150, "Lon": 110.0760, "Elevasi": 5, "Potensi_MW": 15.0, "Investasi_M_IDR": 35.0, "Color": "green", "Icon": "cloud", "APBN_Tahap": "RPJMN 2030-2034", "Waktu_Optimal": "2030-2032"},
    {"Teknologi": "Biomassa", "Kecamatan": "Sentolo", "Lat": -7.7840, "Lon": 110.2220, "Elevasi": 80, "Potensi_MW": 4.5, "Investasi_M_IDR": 12.0, "Color": "darkred", "Icon": "leaf", "APBN_Tahap": "RPJMN 2025-2029", "Waktu_Optimal": "2028-2029"}
])

# ==========================================
# FUNGSI SMART PARSER & MACHINE LEARNING
# ==========================================
@st.cache_data
def process_data_and_predict(file):
    # 1. SMART PARSER: Membaca format NASA JSON, CSV, atau Excel
    if file.name.endswith('.json'):
        import json
        json_data = json.load(file)
        
        # Navigasi ke dalam struktur JSON NASA POWER
        params = json_data.get('properties', {}).get('parameter', {})
        if not params:
            raise ValueError("Format JSON tidak dikenali. Pastikan file berasal dari NASA POWER.")
            
        parsed_data = []
        for param_name, param_values in params.items():
            for date_key, value in param_values.items():
                date_str = str(date_key)
                
                # Tangkap tahun dari 4 digit pertama (Bisa untuk Harian, Bulanan, Tahunan)
                if len(date_str) >= 4:
                    try:
                        year = int(date_str[:4])
                        # Jika format bulanan (6 digit), abaikan bulan ke-13 (NASA menggunakan bulan 13 sebagai rata-rata tahunan)
                        if len(date_str) == 6 and int(date_str[4:]) > 12:
                            continue
                            
                        parsed_data.append({
                            'Tahun': year,
                            'Parameter': param_name,
                            'Value': value if value != -999.0 else np.nan
                        })
                    except ValueError:
                        continue # Abaikan jika key bukan angka
                        
        if len(parsed_data) == 0:
            raise ValueError("Tidak dapat menemukan format waktu yang valid di dalam file JSON.")
            
        df_melted = pd.DataFrame(parsed_data)
        
        # Menghitung rata-rata per tahun secara otomatis
        df_raw = df_melted.pivot_table(index='Tahun', columns='Parameter', values='Value', aggfunc='mean')
        df_raw = df_raw.ffill().bfill() # Bersihkan jika masih ada NaN
        
        # Konversi Parameter NASA ke Daya
        df_clean = pd.DataFrame(index=df_raw.index)
        has_data = False
        if 'ALLSKY_SFC_SW_DWN' in df_raw.columns: 
            df_clean['PLTS (Surya)'] = df_raw['ALLSKY_SFC_SW_DWN'] * 3.5
            has_data = True
        if 'WS10M' in df_raw.columns: 
            df_clean['PLTB (Angin)'] = df_raw['WS10M'] * 2.8
            has_data = True
        if 'PRECTOTCORR' in df_raw.columns: 
            df_clean['PLTMH (Air)'] = df_raw['PRECTOTCORR'] * 1.5
            has_data = True
            
        if not has_data:
            raise ValueError("File JSON tidak mengandung parameter yang dibutuhkan (ALLSKY_SFC_SW_DWN, WS10M, PRECTOTCORR).")
            
        data_input = df_clean

    elif file.name.endswith('.csv'):
        raw_text = file.getvalue().decode("utf-8")
        lines = raw_text.split('\n')
        skip_rows = 0
        for i, line in enumerate(lines):
            if "-END HEADER-" in line or ("YEAR" in line and "MO" in line):
                skip_rows = i + 1 if "-END HEADER-" in line else i
                break
                
        df_raw = pd.read_csv(io.StringIO(raw_text), skiprows=skip_rows)
        if 'YEAR' not in df_raw.columns:
            raise ValueError("Pastikan Anda mengunduh data 'Time Series' dengan kolom 'YEAR', 'MO'.")
            
        df_raw = df_raw.replace(-999.0, np.nan).ffill().bfill()
        
        df_clean = pd.DataFrame()
        df_clean['Tahun'] = df_raw['YEAR']
        has_data = False
        if 'ALLSKY_SFC_SW_DWN' in df_raw.columns: 
            df_clean['PLTS (Surya)'] = df_raw['ALLSKY_SFC_SW_DWN'] * 3.5
            has_data = True
        if 'WS10M' in df_raw.columns: 
            df_clean['PLTB (Angin)'] = df_raw['WS10M'] * 2.8
            has_data = True
        if 'PRECTOTCORR' in df_raw.columns: 
            df_clean['PLTMH (Air)'] = df_raw['PRECTOTCORR'] * 1.5
            has_data = True
            
        if not has_data:
            raise ValueError("File CSV tidak mengandung parameter EBT yang dibutuhkan.")
            
        data_input = df_clean.groupby('Tahun').mean()
        
    else:
        # Untuk format Excel .xlsx / .xls
        data_input = pd.read_excel(file)
        if "Tahun" in data_input.columns:
            data_input = data_input.set_index("Tahun")

    # ==========================================
    # 2. MACHINE LEARNING (Linear Regression Forecasting)
    # ==========================================
    X_train = np.array(data_input.index).reshape(-1, 1)
    tahun_prediksi = np.arange(2025, 2061)
    X_pred = tahun_prediksi.reshape(-1, 1)
    
    data_ml_tahunan = pd.DataFrame(index=tahun_prediksi)
    models = {}
    
    for col in data_input.columns:
        y_train = data_input[col].values
        model = LinearRegression()
        model.fit(X_train, y_train)
        models[col] = model
        
        tren_prediksi = model.predict(X_pred)
        noise = np.random.normal(0, np.std(y_train) * 0.5, len(tahun_prediksi))
        data_ml_tahunan[col] = np.maximum(0, tren_prediksi + noise)

    return data_input, data_ml_tahunan

# ==========================================
# MAIN CONTENT
# ==========================================
if uploaded_file is not None:
    try:
        # Eksekusi AI Data Processing
        data_historis, data_ml_tahunan = process_data_and_predict(uploaded_file)
        
        # Hitung Rata-rata 5 Tahunan
        # Resampling 5 tahun dari indeks tahun bulat
        data_5_tahun = data_ml_tahunan.groupby(data_ml_tahunan.index // 5 * 5).mean()

        tab1, tab2, tab3 = st.tabs(["📈 Prediksi ML (Scikit-Learn)", "⚖️ Keputusan MCDM", "🏔️ Terrain Spasial 3D"])

        # ---------------- TAB 1: PREDIKSI ML ----------------
        with tab1:
            st.header(f"Proyeksi Suplai Energi Berbasis Regresi Linear (hingga 2060)")
            
            # Cek jika data dari NASA, tampilkan notifikasi
            if uploaded_file.name.endswith('.csv'):
                st.success("🛰️ **NASA POWER Data Detected!** Sistem secara otomatis memotong metadata, mengatasi *missing values* (-999.0), dan menormalisasi *irradiance/wind speed* menjadi ekuivalen kapasitas daya.")
            
            st.write("**Historis vs Prediksi (MW):**")
            # Menggabungkan historis dan prediksi untuk grafik
            df_gabung = pd.concat([data_historis, data_ml_tahunan])
            fig_gabung = px.line(
                df_gabung.reset_index().melt(id_vars="index", var_name="Teknologi", value_name="MW"),
                x="index", y="MW", color="Teknologi",
                labels={"index": "Tahun"}, template="plotly_dark",
                title="Histori Data (Kiri) & Tren Prediksi ML (Kanan)"
            )
            # Garis pembatas tahun sekarang
            fig_gabung.add_vline(x=2024, line_dash="dash", line_color="red", annotation_text="Mulai Prediksi")
            st.plotly_chart(fig_gabung, use_container_width=True)

        # ---------------- TAB 2: MCDM ----------------
        with tab2:
            st.header("MCDM dengan Parameter Dinamis")
            n_alt = len(data_historis.columns)
            capex_awal    = np.array([12.0, 18.0, 22.0, 25.0][:n_alt])
            emisi_aktual  = np.array([40.0, 11.0, 24.0, 230.0][:n_alt])
            sosial_aktual = np.array([90, 70, 85, 80][:n_alt])
            capex_terinflasi = capex_awal * ((1 + inflasi) ** selisih_tahun)

            if kebijakan == "Pajak Karbon Tinggi (Pro-Lingkungan)":
                pengali_emisi = 1.5
            elif kebijakan == "Subsidi Masif EBT (Pro-Ekonomi)":
                pengali_emisi = 1.0
                capex_terinflasi *= 0.7
            else:
                pengali_emisi = 1.0

            col_b1, col_b2 = st.columns([1, 2])
            with col_b1:
                w_tech = st.slider("Potensi Daya (Benefit)", 0.0, 1.0, 0.4)
                w_econ = st.slider("Biaya Investasi (Cost)", 0.0, 1.0, 0.3)
                w_env  = st.slider("Emisi Karbon (Cost)", 0.0, 1.0, 0.2) * pengali_emisi
                w_soc  = st.slider("Penerimaan Sosial (Benefit)", 0.0, 1.0, 0.1)
                total  = w_tech + w_econ + w_env + w_soc
                if total == 0: st.stop()
                w_tech, w_econ, w_env, w_soc = w_tech/total, w_econ/total, w_env/total, w_soc/total

            with col_b2:
                # Ambil skor daya sesuai slider tahun_evaluasi
                skor_daya = data_ml_tahunan.loc[tahun_evaluasi] if tahun_evaluasi in data_ml_tahunan.index else data_ml_tahunan.iloc[-1]
                
                data_aktual = pd.DataFrame({
                    "Prediksi Daya (MW)": skor_daya.values,
                    "Investasi (M IDR/MW)": capex_terinflasi,
                    "Emisi (Ton/GWh)": emisi_aktual,
                    "Sosial (1-100)": sosial_aktual,
                }, index=data_historis.columns)
                st.dataframe(data_aktual.style.format("{:.2f}"))

            # Normalisasi & Kalkulasi MCDM
            norm_df = pd.DataFrame(index=data_aktual.index)
            norm_df["Daya"] = data_aktual["Prediksi Daya (MW)"] / data_aktual["Prediksi Daya (MW)"].max()
            norm_df["Sosial"] = data_aktual["Sosial (1-100)"] / data_aktual["Sosial (1-100)"].max()
            norm_df["Investasi"] = data_aktual["Investasi (M IDR/MW)"].min() / data_aktual["Investasi (M IDR/MW)"]
            norm_df["Emisi"] = data_aktual["Emisi (Ton/GWh)"].min() / data_aktual["Emisi (Ton/GWh)"]

            skor_akhir = (norm_df["Daya"]*w_tech + norm_df["Investasi"]*w_econ + norm_df["Emisi"]*w_env + norm_df["Sosial"]*w_soc)
            hasil_df = pd.DataFrame(skor_akhir, columns=["Skor"]).sort_values("Skor", ascending=False)

            st.divider()
            st.subheader(f"🏆 Rekomendasi Prioritas Tahun {tahun_evaluasi}")
            fig_rank = px.bar(hasil_df.reset_index(), x="Skor", y="index", orientation="h", color="Skor", template="plotly_dark")
            fig_rank.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_rank, use_container_width=True)

            # ==========================================
            # GENERATIVE AI INSIGHT
            # ==========================================
            st.divider()
            st.subheader("🤖 AI Executive Summary")
            
            if st.button("✨ Generate AI Insight"):
                if not api_key:
                    st.warning("⚠️ Masukkan API Key.")
                else:
                    try:
                        client = genai.Client(api_key=api_key)
                        prompt_ai = f"""
Anda adalah pakar transisi energi. Berikan ringkasan berformat JSON (maks 4 kalimat per poin):
{{ "mengapa_menang": "...", "dampak_ekonomi": "...", "rekomendasi": "..." }}
Data: Tahun {tahun_evaluasi}, Kebijakan {kebijakan}, Inflasi {inflasi*100}%. 
Ranking: {hasil_df.to_string()}
Jawab murni JSON.
"""
                        with st.spinner("AI menganalisis..."):
                            response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt_ai)
                            raw = response.text.strip()
                            if raw.startswith("```"):
                                raw = raw.split("```")[1]
                                if raw.startswith("json"): raw = raw[4:]
                            st.session_state.ai_result_text = json.loads(raw)
                    except Exception as e:
                        st.error(f"Error AI: {e}")

            if st.session_state.ai_result_text:
                res = st.session_state.ai_result_text
                c1, c2, c3 = st.columns(3)
                c1.info(f"🏆 **Mengapa Menang?**\n\n{res.get('mengapa_menang','')}")
                c2.warning(f"💰 **Dampak Ekonomi**\n\n{res.get('dampak_ekonomi','')}")
                c3.success(f"📋 **Rekomendasi**\n\n{res.get('rekomendasi','')}")

        # ---------------- TAB 3: TERRAIN 3D ----------------
        with tab3:
            st.header("🏔️ Digital Twin: Terrain Mode Kulon Progo")
            st.write("Visualisasi topografi nyata. Tinggi kolom = Potensi Daya. Semakin kasar teksturnya, semakin tinggi elevasinya (pegunungan).")

            df_pdk = df_lokasi.copy()
            color_pdk = {"blue": [41, 182, 246, 200], "orange": [255, 167, 38, 200], "green": [102, 187, 106, 200], "darkred": [239, 83, 80, 200]}
            df_pdk["color_rgb"] = df_pdk["Color"].map(color_pdk)

            view_state = pdk.ViewState(latitude=-7.8288, longitude=110.1587, zoom=10.5, pitch=50, bearing=-10)

            column_layer = pdk.Layer(
                "ColumnLayer",
                data=df_pdk,
                get_position=["Lon", "Lat"],
                get_elevation="Potensi_MW * 200",
                elevation_scale=1, radius=400,
                get_fill_color="color_rgb",
                pickable=True, auto_highlight=True,
            )

            terrain_layer = pdk.Layer(
                "TerrainLayer",
                elevation_decoder={"rScaler": 1, "gScaler": 0, "bScaler": 0, "offset": 0},
                elevation_data="https://assets.mapbox.com/raster-tiles/mapbox.terrain-rgb/{z}/{x}/{y}.pngraw",
                texture="https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
            )

            r = pdk.Deck(
                layers=[terrain_layer, column_layer],
                initial_view_state=view_state,
                tooltip={"html": "<b>{Teknologi}</b><br/>📍 {Kecamatan}<br/>⚡ {Potensi_MW} MW", "style": {"backgroundColor": "steelblue", "color": "white"}}
            )
            st.pydeck_chart(r)

    except Exception as e:
        st.error(f"Terjadi kesalahan pemrosesan: {e}")
else:
    st.info("👈 Upload file Excel atau CSV NASA POWER untuk memulai.")
