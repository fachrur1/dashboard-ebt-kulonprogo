import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
try:
    from google import genai
except ImportError:
    st.error("Library google-genai belum terinstall. Silakan install via pip.")
    st.stop()
import plotly.express as px
import plotly.graph_objects as go
import json
import io
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Dashboard EBT Kulon Progo - Oji", layout="wide")

st.title("⚡ Dashboard Analisis Potensi EBT & MCDM Kabupaten Kulon Progo")
st.markdown("**Oleh: Muhammad Fachrurrozy (Teknik Fisika UGM)**")
st.divider()

# ==========================================
# SESSION STATE (Menyimpan Status)
# ==========================================
if "ai_result_text" not in st.session_state: st.session_state.ai_result_text = None
if "ai_result_model" not in st.session_state: st.session_state.ai_result_model = None
if "ai_chart_data" not in st.session_state: st.session_state.ai_chart_data = None
if "use_dummy" not in st.session_state: st.session_state.use_dummy = False
if "data_dummy" not in st.session_state: st.session_state.data_dummy = None

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.header("📂 1. Input Data Sistem")
st.sidebar.caption("Support: Excel (.xlsx), NASA POWER (.csv/.json)")

# Tombol untuk generate data dummy
if st.sidebar.button("🎲 Generate Contoh Data Dummy"):
    np.random.seed(42)
    years = list(range(2015, 2025))
    data_dummy = {
        'Tahun': years,
        'PLTS (Surya)': [15.2, 15.5, 15.8, 16.1, 16.3, 16.6, 16.9, 17.2, 17.4, 17.7],
        'PLTB (Angin)': [12.1, 12.3, 12.4, 12.6, 12.7, 12.9, 13.0, 13.2, 13.3, 13.5],
        'PLTMH (Air)': [18.2, 18.3, 18.4, 18.5, 18.5, 18.6, 18.7, 18.8, 18.8, 18.9],
        'Biomassa': [5.0, 5.1, 5.2, 5.0, 5.3, 5.1, 5.2, 5.4, 5.2, 5.5]
    }
    st.session_state['use_dummy'] = True
    st.session_state['data_dummy'] = pd.DataFrame(data_dummy)
    st.sidebar.success("✅ Data dummy berhasil dimuat!")

st.sidebar.divider()
uploaded_file = st.sidebar.file_uploader("Upload File Data", type=["xlsx", "xls", "csv", "json"])

if uploaded_file is not None:
    st.session_state['use_dummy'] = False

st.sidebar.header("🌍 2. Asumsi Makro & Kebijakan")
inflasi = st.sidebar.number_input("Tingkat Inflasi Tahunan (%)", min_value=0.0, max_value=15.0, value=3.5, step=0.1) / 100
kebijakan = st.sidebar.selectbox("Skenario Kebijakan", ["Business as Usual (BAU)", "Pajak Karbon Tinggi (Pro-Lingkungan)", "Subsidi Masif EBT (Pro-Ekonomi)"])

# SLIDER INI SEKARANG MENJADI KUNCI SINKRONISASI SEMUA TAB
tahun_evaluasi = st.sidebar.slider("Tahun Target Evaluasi (Time Machine)", 2025, 2060, 2060)
selisih_tahun = tahun_evaluasi - 2025
st.sidebar.divider()

st.sidebar.header("🧠 3. Integrasi AI")
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    st.sidebar.success("✅ AI terhubung via Server Key.")
except Exception:
    api_key = st.sidebar.text_input("Gemini API Key", type="password", placeholder="Masukkan API Key...")

# ==========================================
# DATA LOKASI EBT
# ==========================================
LOKASI_EBT = [
    {"Teknologi": "PLTMH (Air)", "Kecamatan": "Girimulyo", "Lat": -7.7470, "Lon": 110.1260, "Elevasi": 350, "Potensi_MW": 8.5, "Investasi_M_IDR": 18.5, "Color": "blue", "Icon": "tint", "APBN_Tahap": "RPJMN 2025-2029", "Waktu_Optimal": "2026-2027", "Alasan": "Debit sungai tinggi, infrastruktur jalan sudah ada."},
    {"Teknologi": "PLTMH (Air)", "Kecamatan": "Samigaluh", "Lat": -7.6710, "Lon": 110.1700, "Elevasi": 420, "Potensi_MW": 6.2, "Investasi_M_IDR": 14.0, "Color": "blue", "Icon": "tint", "APBN_Tahap": "RPJMN 2025-2029", "Waktu_Optimal": "2027-2028", "Alasan": "Perlu studi hidrologi lanjutan, akses jalan terbatas."},
    {"Teknologi": "PLTS (Surya)", "Kecamatan": "Wates", "Lat": -7.8600, "Lon": 110.1400, "Elevasi": 15, "Potensi_MW": 22.0, "Investasi_M_IDR": 28.0, "Color": "orange", "Icon": "sun", "APBN_Tahap": "RPJMN 2025-2029", "Waktu_Optimal": "2025-2026", "Alasan": "Irradiasi matahari tinggi, lahan tersedia, dekat jaringan PLN."},
    {"Teknologi": "PLTB (Angin)", "Kecamatan": "Pantai Glagah", "Lat": -7.9150, "Lon": 110.0760, "Elevasi": 5, "Potensi_MW": 15.0, "Investasi_M_IDR": 35.0, "Color": "green", "Icon": "cloud", "APBN_Tahap": "RPJMN 2030-2034", "Waktu_Optimal": "2030-2032", "Alasan": "Kecepatan angin laut stabil >5 m/s. Butuh kajian lingkungan pesisir."},
    {"Teknologi": "Biomassa", "Kecamatan": "Sentolo", "Lat": -7.7840, "Lon": 110.2220, "Elevasi": 80, "Potensi_MW": 4.5, "Investasi_M_IDR": 12.0, "Color": "darkred", "Icon": "leaf", "APBN_Tahap": "RPJMN 2025-2029", "Waktu_Optimal": "2028-2029", "Alasan": "Limbah pertanian melimpah, perlu kemitraan petani lokal."},
]
df_lokasi = pd.DataFrame(LOKASI_EBT)

# ==========================================
# FUNGSI SMART PARSER & MACHINE LEARNING
# ==========================================
@st.cache_data
def process_data_and_predict(file_bytes, file_name, tahun_akhir=2060, random_seed=42):
    np.random.seed(random_seed)

    if file_name.endswith('.json'):
        raw_json = json.loads(file_bytes.decode("utf-8"))
        is_nasa_geojson = isinstance(raw_json, dict) and raw_json.get("type") == "Feature" and "properties" in raw_json and "parameter" in raw_json.get("properties", {})

        if is_nasa_geojson:
            NASA_MAP = {"ALLSKY_SFC_SW_DWN": ("PLTS (Surya)", 3.5), "WS10M": ("PLTB (Angin)", 2.8), "PRECTOTCORR": ("PLTMH (Air)", 1.5)}
            fill_val = raw_json.get("header", {}).get("fill_value", -999)
            params = raw_json["properties"]["parameter"]
            sample_param = list(params.values())[0]
            is_climatology = any(k in ["JAN", "FEB", "ANN"] for k in sample_param.keys())
            year_data = {}

            if is_climatology:
                for param_key, monthly in params.items():
                    if param_key not in NASA_MAP: continue
                    col_name, factor = NASA_MAP[param_key]
                    if "ANN" in monthly and monthly["ANN"] != fill_val: ann_val = monthly["ANN"] * factor
                    else:
                        valid_vals = [v for k, v in monthly.items() if k not in ["ANN", "DJF", "MAM", "JJA", "SON"] and v != fill_val]
                        ann_val = np.mean(valid_vals) * factor if valid_vals else 0
                    for yr in range(2015, 2025):
                        year_data.setdefault(yr, {})[col_name] = max(0, ann_val + np.random.normal(0, ann_val * 0.05))
            else:
                for param_key, monthly in params.items():
                    if param_key not in NASA_MAP: continue
                    col_name, factor = NASA_MAP[param_key]
                    for k, v in monthly.items():
                        try: year, month = int(k[:4]), int(k[4:])
                        except (ValueError, IndexError): continue
                        if month == 13 or v == fill_val or v is None: continue
                        year_data.setdefault(year, {}).setdefault(col_name, []).append(v * factor)
                for yr in year_data:
                    for col in year_data[yr]:
                        if isinstance(year_data[yr][col], list): year_data[yr][col] = float(np.mean(year_data[yr][col]))

            if not year_data: raise ValueError("Data JSON NASA tidak mengandung parameter EBT.")
            records = [{"Tahun": yr, **cols} for yr, cols in sorted(year_data.items())]
            data_input = pd.DataFrame.from_records(records).set_index("Tahun")

        elif isinstance(raw_json, list):
            df_json = pd.DataFrame.from_records(raw_json).rename(columns=lambda x: "Tahun" if x.strip().lower() in ["tahun", "year"] else x)
            df_json["Tahun"] = df_json["Tahun"].astype(int)
            data_input = df_json.set_index("Tahun").apply(pd.to_numeric, errors='coerce').dropna(how="all")

        elif isinstance(raw_json, dict):
            df_json = pd.DataFrame({k: pd.Series(v) for k, v in raw_json.items()}).rename(columns=lambda x: "Tahun" if x.strip().lower() in ["tahun", "year"] else x)
            df_json["Tahun"] = df_json["Tahun"].astype(int)
            data_input = df_json.set_index("Tahun").apply(pd.to_numeric, errors='coerce').dropna(how="all")

    elif file_name.endswith('.csv'):
        raw_text = file_bytes.decode("utf-8")
        skip_rows = next((i + 1 if "-END HEADER-" in line else i for i, line in enumerate(raw_text.split('\n')) if "-END HEADER-" in line or ("YEAR" in line and "MO" in line)), 0)
        df_raw = pd.read_csv(io.StringIO(raw_text), skiprows=skip_rows).replace(-999.0, np.nan).ffill().bfill()
        df_clean = pd.DataFrame({'Tahun': df_raw['YEAR'].astype(int)})
        
        conversion_factors = {'ALLSKY_SFC_SW_DWN': ('PLTS (Surya)', 3.5), 'WS10M': ('PLTB (Angin)', 2.8), 'PRECTOTCORR': ('PLTMH (Air)', 1.5)}
        for nasa_col, (new_col, factor) in conversion_factors.items():
            if nasa_col in df_raw.columns: df_clean[new_col] = df_raw[nasa_col] * factor
        
        if len(df_clean.columns) == 1: 
            if 'PLTS (Surya)' in df_raw.columns: df_clean = df_raw.copy()
        data_input = df_clean.groupby('Tahun').mean()

    else:
        data_input = pd.read_excel(io.BytesIO(file_bytes))
        col_map = {c: "Tahun" if c.lower() == "tahun" else c for c in data_input.columns}
        data_input = data_input.rename(columns=col_map).set_index("Tahun")

    data_input.index = data_input.index.astype(int)
    tahun_prediksi = np.arange(2025, tahun_akhir + 1)
    X_train = data_input.index.values.reshape(-1, 1).astype(float)
    X_pred = tahun_prediksi.reshape(-1, 1).astype(float)

    data_ml_tahunan = pd.DataFrame(index=pd.Index(tahun_prediksi, name='Tahun'))
    range_bulan = pd.date_range(start="2025-01-01", end=f"{tahun_akhir}-12-31", freq="MS")
    data_ml_bulan = pd.DataFrame(index=range_bulan)
    metrics = {}
    rng = np.random.default_rng(seed=random_seed)

    profil_musiman = {
        "surya": {1:0.85, 2:0.85, 3:0.95, 4:1.00, 5:1.10, 6:1.15, 7:1.20, 8:1.25, 9:1.15, 10:1.05, 11:0.90, 12:0.85},
        "angin": {1:0.80, 2:0.80, 3:0.90, 4:0.95, 5:1.10, 6:1.20, 7:1.30, 8:1.35, 9:1.20, 10:1.00, 11:0.90, 12:0.80},
        "air":   {1:1.30, 2:1.35, 3:1.20, 4:1.10, 5:0.90, 6:0.70, 7:0.50, 8:0.40, 9:0.45, 10:0.70, 11:1.10, 12:1.25}
    }

    for col in data_input.columns:
        y_train = data_input[col].values.astype(float)
        base_val = float(np.mean(y_train))
        
        if np.std(y_train) == 0:
            data_ml_tahunan[col] = base_val
            data_ml_bulan[col] = base_val
            metrics[col] = {"R²": 1.0, "MAE": 0.0}
            continue
            
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred_hist = model.predict(X_train)
        metrics[col] = {"R²": round(r2_score(y_train, y_pred_hist), 3), "MAE": round(mean_absolute_error(y_train, y_pred_hist), 3)}

        tren_tahunan_mentah = model.predict(X_pred)
        val_akhir = tren_tahunan_mentah[-1]
        batas_atas = base_val * 1.20
        batas_bawah = base_val * 0.80

        if val_akhir > batas_atas or val_akhir < batas_bawah:
            target_val = batas_atas if val_akhir > batas_atas else batas_bawah
            start_val = model.predict([[tahun_prediksi[0]]])[0]
            slope_baru = (target_val - start_val) / len(tahun_prediksi)
            tren_tahunan = start_val + slope_baru * np.arange(len(tahun_prediksi))
            tahun_bulan = range_bulan.year.values.astype(float).reshape(-1, 1)
            tren_bulan = start_val + slope_baru * (tahun_bulan[:, 0] - tahun_prediksi[0])
        else:
            tren_tahunan = tren_tahunan_mentah
            tahun_bulan = range_bulan.year.values.astype(float).reshape(-1, 1)
            tren_bulan = model.predict(tahun_bulan)

        kunci_profil = "default"
        if "surya" in col.lower() or "plts" in col.lower(): kunci_profil = "surya"
        elif "angin" in col.lower() or "pltb" in col.lower(): kunci_profil = "angin"
        elif "air" in col.lower() or "pltmh" in col.lower(): kunci_profil = "air"
        
        pengali_musiman = np.array([profil_musiman[kunci_profil][m] for m in range_bulan.month]) if kunci_profil != "default" else np.ones(len(range_bulan))

        noise_std = np.std(y_train) * 0.6 if np.std(y_train) > 0 else base_val * 0.1
        data_ml_tahunan[col] = np.maximum(base_val * 0.2, tren_tahunan + rng.normal(0, noise_std, len(tahun_prediksi)))
        data_ml_bulan[col] = np.maximum(base_val * 0.1, (tren_bulan * pengali_musiman) + rng.normal(0, base_val * 0.05, len(range_bulan)))

    return data_input, data_ml_tahunan, data_ml_bulan, metrics

def get_prediction_for_year(data_ml: pd.DataFrame, tahun: int) -> pd.Series:
    if tahun in data_ml.index: return data_ml.loc[tahun]
    return data_ml.loc[data_ml.index[np.argmin(np.abs(data_ml.index - tahun))]]

# ==========================================
# MAIN CONTENT
# ==========================================
if uploaded_file is not None or st.session_state.use_dummy:
    try:
        if st.session_state.use_dummy and st.session_state.data_dummy is not None:
            file_name = "data_dummy.csv"
            file_bytes = st.session_state.data_dummy.to_csv(index=False).encode('utf-8')
        else:
            file_bytes = uploaded_file.getvalue()
            file_name = uploaded_file.name

        # 1. LATIH ML SATU KALI HINGGA 2060
        data_historis_full, data_ml_tahunan_full, data_ml_bulan_full, metrics = process_data_and_predict(file_bytes, file_name, tahun_akhir=2060)

        # 2. POTONG (SLICE) DATA BERDASARKAN SLIDER TAHUN_EVALUASI
        data_ml_tahunan = data_ml_tahunan_full[data_ml_tahunan_full.index <= tahun_evaluasi]
        cols_teknologi = list(data_historis_full.columns)

        # Siapkan Bar Chart 5 Tahunan
        ml_copy = data_ml_tahunan.copy()
        ml_copy['Tahun Awal'] = (ml_copy.index // 5) * 5
        ml_copy['Periode'] = ml_copy['Tahun Awal'].astype(str) + "–" + (ml_copy['Tahun Awal'] + 4).astype(str)
        data_5_tahun = ml_copy.groupby('Periode')[cols_teknologi].mean()

        # Konfigurasi Parameter MCDM Dasar
        n_alt = len(cols_teknologi)
        base_capex = [12.0, 18.0, 22.0, 25.0, 30.0][:n_alt]
        base_emisi = [40.0, 11.0, 24.0, 230.0, 60.0][:n_alt]
        base_sosial = [90, 70, 85, 80, 75][:n_alt]
        
        capex_awal = np.array(base_capex)
        emisi_aktual = np.array(base_emisi)
        sosial_aktual = np.array(base_sosial)

        # TAB LAYOUT
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Analisis Prediksi (ML)", "⚖️ MCDM & Kebijakan", "🗺️ Peta Potensi Spasial", "🧠 Deep Dive (Skripsi)"])

        # ---------------- TAB 1: PREDIKSI ML ----------------
        with tab1:
            st.header(f"Proyeksi Potensi Daya (2015 hingga {tahun_evaluasi})")
            if file_name.endswith('.json'): st.success("📋 **JSON Terdeteksi!** Modul pembaca telah diperbarui.")

            st.subheader("📊 Evaluasi Kualitas Model (Linear Damped Regression)")
            st.dataframe(pd.DataFrame(metrics).T.style.format("{:.3f}").background_gradient(subset=["R²"], cmap="RdYlGn").background_gradient(subset=["MAE"], cmap="RdYlGn_r"), use_container_width=True)
            st.divider()

            df_hist = data_historis_full.copy().reset_index()
            df_hist['Tipe'] = 'Historis'
            df_pred = data_ml_tahunan.copy().reset_index()
            df_pred['Tipe'] = 'Prediksi'

            df_gabung = pd.concat([df_hist[['Tahun'] + cols_teknologi + ['Tipe']], df_pred[['Tahun'] + cols_teknologi + ['Tipe']]], ignore_index=True)
            df_melted = df_gabung.melt(id_vars=['Tahun', 'Tipe'], value_vars=cols_teknologi, var_name='Teknologi', value_name='MW')

            fig_tren = px.line(df_melted, x='Tahun', y='MW', color='Teknologi', line_dash='Tipe', template='plotly_white', markers=True, title=f"Histori Data vs Tren Prediksi ML (hingga {tahun_evaluasi})")
            fig_tren.add_vline(x=2025, line_dash='dot', line_color='red', annotation_text="Mulai Prediksi")
            fig_tren.update_layout(hovermode='x unified', yaxis_title="Potensi (MW)", xaxis_title="Tahun")
            st.plotly_chart(fig_tren, use_container_width=True)

            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.write(f"**Tren Rata-rata Per 5 Tahun:**")
                fig_bar = px.bar(data_5_tahun.reset_index().melt(id_vars="Periode", value_vars=cols_teknologi, var_name="Teknologi", value_name="MW"), x="Periode", y="MW", color="Teknologi", barmode="group", template="plotly_white")
                fig_bar.update_layout(yaxis_title="Potensi (MW)", xaxis_title="Periode 5 Tahunan")
                st.plotly_chart(fig_bar, use_container_width=True)

            with col_c2:
                st.write(f"**Fluktuasi Musiman Spesifik (Tahun {tahun_evaluasi}):**")
                data_tahun_terakhir = data_ml_bulan_full[data_ml_bulan_full.index.year == tahun_evaluasi].copy()
                if not data_tahun_terakhir.empty:
                    data_tahun_terakhir['Bulan'] = data_tahun_terakhir.index.strftime('%B')
                    fig_line = px.line(data_tahun_terakhir.melt(id_vars=['Bulan'], value_vars=cols_teknologi, var_name='Teknologi', value_name='MW'), x="Bulan", y="MW", color="Teknologi", template="plotly_white", markers=True)
                    fig_line.update_xaxes(categoryorder='array', categoryarray=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
                    fig_line.update_layout(yaxis_title="Potensi (MW)", xaxis_title="Bulan")
                    st.plotly_chart(fig_line, use_container_width=True)

        # ---------------- TAB 2: MCDM ----------------
        with tab2:
            st.header(f"MCDM & Analisis Keputusan (Target: {tahun_evaluasi})")
            capex_terinflasi = capex_awal.copy() * ((1 + inflasi) ** selisih_tahun)

            if kebijakan == "Pajak Karbon Tinggi (Pro-Lingkungan)":
                pengali_emisi = 1.5
                st.warning("🌿 Penalti emisi diaktifkan (+50% bobot emisi).")
            elif kebijakan == "Subsidi Masif EBT (Pro-Ekonomi)":
                pengali_emisi = 1.0
                capex_terinflasi *= 0.7
                st.info("💰 Diskon investasi 30% diaktifkan.")
            else:
                pengali_emisi = 1.0

            col_b1, col_b2 = st.columns([1, 2])
            with col_b1:
                w_tech = st.slider("Potensi Daya (Benefit)", 0.0, 1.0, 0.4, step=0.05)
                w_econ = st.slider("Biaya Investasi (Cost)", 0.0, 1.0, 0.3, step=0.05)
                w_env = st.slider("Emisi Karbon (Cost)", 0.0, 1.0, 0.2, step=0.05) * pengali_emisi
                w_soc = st.slider("Penerimaan Sosial (Benefit)", 0.0, 1.0, 0.1, step=0.05)
                total = w_tech + w_econ + w_env + w_soc
                if total == 0: st.stop()
                w_tech, w_econ, w_env, w_soc = w_tech/total, w_econ/total, w_env/total, w_soc/total
                st.metric("Total Bobot (Setelah Normalisasi)", f"{total:.2f} → 1.00")

            with col_b2:
                skor_daya = get_prediction_for_year(data_ml_tahunan_full, tahun_evaluasi)
                data_aktual = pd.DataFrame({
                    "Daya Prediksi (MW)": skor_daya.values,
                    "Investasi Terinflasi (M IDR/MW)": capex_terinflasi,
                    "Emisi (Ton CO2e/GWh)": emisi_aktual,
                    "Sosial (1-100)": sosial_aktual,
                }, index=cols_teknologi)
                
                st.write("**1. Tabel Data Aktual/Raw (Tidak Berubah oleh Bobot)**")
                st.dataframe(data_aktual.style.format("{:.2f}").background_gradient(cmap="Blues"), use_container_width=True)

            # Normalisasi MCDM
            norm_df = pd.DataFrame(index=data_aktual.index)
            norm_df["Daya"] = data_aktual["Daya Prediksi (MW)"] / data_aktual["Daya Prediksi (MW)"].max()
            norm_df["Sosial"] = data_aktual["Sosial (1-100)"] / data_aktual["Sosial (1-100)"].max()
            norm_df["Investasi"] = data_aktual["Investasi Terinflasi (M IDR/MW)"].min() / data_aktual["Investasi Terinflasi (M IDR/MW)"]
            norm_df["Emisi"] = data_aktual["Emisi (Ton CO2e/GWh)"].min() / data_aktual["Emisi (Ton CO2e/GWh)"]

            bobot_label = {"Daya": w_tech, "Investasi": w_econ, "Emisi": w_env, "Sosial": w_soc}
            categories = list(bobot_label.keys())
            weighted_df = pd.DataFrame({dim: norm_df[dim] * bobot_label[dim] for dim in categories}, index=norm_df.index)

            st.write("**2. Tabel Data Normalisasi Terbobot (Berubah Mengikuti Slider User)**")
            st.dataframe(weighted_df.style.format("{:.3f}").background_gradient(cmap="Greens"), use_container_width=True)

            skor_akhir = weighted_df.sum(axis=1)
            hasil_df = pd.DataFrame(skor_akhir, columns=["Skor Preferensi"]).sort_values("Skor Preferensi", ascending=False)

            st.divider()
            st.subheader(f"🏆 Rekomendasi Prioritas Tahun {tahun_evaluasi}")
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                fig_rank = px.bar(hasil_df.reset_index().rename(columns={"index": "Teknologi"}), x="Skor Preferensi", y="Teknologi", orientation="h", color="Skor Preferensi", color_continuous_scale="Teal", template="plotly_white")
                fig_rank.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_title="Skor Preferensi")
                st.plotly_chart(fig_rank, use_container_width=True)

            with col_r2:
                st.success(f"**Pemenang Utama:** {hasil_df.index[0]} | Skor: {hasil_df.iloc[0]['Skor Preferensi']:.3f}")

                fig_radar = go.Figure()
                for alt in weighted_df.index:
                    vals = weighted_df.loc[alt, categories].tolist()
                    fig_radar.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=[f"{d}" for d in categories] + [categories[0]], fill='toself', name=alt))
                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, max(w_tech, w_econ, w_env, w_soc) * 1.2])), template="plotly_white", height=400, title="Analisis Radar Terbobot")
                st.plotly_chart(fig_radar, use_container_width=True)

        # ---------------- TAB 3: PETA 3D ----------------
        with tab3:
            st.header("🗺️ Pemetaan Geospasial Potensi EBT Kulon Progo")
            mode_peta = st.radio("Mode Tampilan Peta:", ["🗺️ 2D Interaktif", "🏔️ 3D Globe (Elevasi)"], horizontal=True)

            if mode_peta == "🗺️ 2D Interaktif":
                m = folium.Map(location=[-7.8288, 110.1587], zoom_start=11, tiles="CartoDB dark_matter")
                for _, row in df_lokasi.iterrows():
                    popup_html = f"<div style='width: 200px;'><b>{row['Teknologi']}</b><br>📍 {row['Kecamatan']}<br>⚡ {row['Potensi_MW']} MW<br>💰 Rp {row['Investasi_M_IDR']} M<br>📅 {row['Waktu_Optimal']}</div>"
                    folium.Marker([row['Lat'], row['Lon']], popup=folium.Popup(popup_html, max_width=260), icon=folium.Icon(color=row['Color'], icon=row['Icon'], prefix='fa')).add_to(m)
                st_folium(m, width=None, height=500, use_container_width=True)
            else:
                color_map = {"PLTMH (Air)": "#29b6f6", "PLTS (Surya)": "#ffa726", "PLTB (Angin)": "#66bb6a", "Biomassa": "#ef5350"}
                fig_3d = go.Figure()
                for _, row in df_lokasi.iterrows():
                    fig_3d.add_trace(go.Scatter3d(x=[row['Lon']], y=[row['Lat']], z=[row['Elevasi']], mode='markers+text', marker=dict(size=row['Potensi_MW'] * 1.2, color=color_map.get(row['Teknologi'], "#fff"), opacity=0.85, line=dict(width=0.5, color='white')), text=[f"<b>{row['Teknologi']}</b><br>{row['Potensi_MW']} MW"], textposition="top center", name=row['Teknologi']))
                fig_3d.update_layout(scene=dict(bgcolor="rgb(10,15,30)", xaxis=dict(gridcolor="#1e3a5f", title="Longitude"), yaxis=dict(gridcolor="#1e3a5f", title="Latitude"), zaxis=dict(gridcolor="#1e3a5f", title="Elevasi (m)")), template="plotly_dark", height=600)
                st.plotly_chart(fig_3d, use_container_width=True)

        # ---------------- TAB 4: DEEP DIVE (6 GRAFIK BARU) ----------------
        with tab4:
            st.header("🧠 Deep Dive Metodologi Analisis (Skripsi)")
            st.caption("Penjelasan komprehensif mengenai efek statistik, ekonomi, dan lingkungan yang membangun hasil di Tab sebelumnya.")
            
            c_deep1, c_deep2 = st.columns(2)
            
            with c_deep1:
                # Grafik 1: Boxplot Volatilitas Historis
                df_hist_melt = data_historis_full.reset_index().melt(id_vars="Tahun", value_vars=cols_teknologi, var_name="Teknologi", value_name="MW")
                fig_box = px.box(df_hist_melt, x="Teknologi", y="MW", color="Teknologi", title="1. Distribusi & Volatilitas Historis (Boxplot)", template="plotly_white")
                fig_box.update_layout(showlegend=False)
                st.plotly_chart(fig_box, use_container_width=True)
                st.info("💡 **Makna:** Mengukur seberapa stabil sumber energi di masa lalu. Box yang panjang (rentang lebar) menandakan volatilitas iklim yang tinggi pada teknologi tersebut.")

                # Grafik 3: Proyeksi Pembengkakan Investasi
                tahun_range = np.arange(2025, tahun_evaluasi + 1)
                capex_history = {tech: [capex_awal[i] * ((1 + inflasi) ** (y - 2025)) for y in tahun_range] for i, tech in enumerate(cols_teknologi)}
                fig_capex = px.line(pd.DataFrame(capex_history, index=tahun_range).reset_index().melt(id_vars="index", var_name="Teknologi", value_name="CAPEX"), x="index", y="CAPEX", color="Teknologi", title="3. Proyeksi Efek Inflasi terhadap Investasi", template="plotly_white")
                fig_capex.update_layout(xaxis_title="Tahun", yaxis_title="Biaya Investasi (M IDR / MW)")
                st.plotly_chart(fig_capex, use_container_width=True)
                st.info(f"💡 **Makna:** Membuktikan efek finansial dari compound interest (inflasi {inflasi*100}%). Menunda proyek hingga tahun {tahun_evaluasi} akan membengkakkan biaya modal secara eksponensial.")

                # Grafik 5: Dekomposisi MCDM
                fig_stacked = px.bar(weighted_df.reset_index(), x="index", y=categories, title="5. Dekomposisi Skor MCDM (Efek Bobot)", template="plotly_white")
                fig_stacked.update_layout(xaxis_title="Teknologi", yaxis_title="Skor Normalisasi Terbobot", barmode='stack')
                st.plotly_chart(fig_stacked, use_container_width=True)
                st.info("💡 **Makna:** Menjabarkan darimana skor total setiap alternatif berasal. Bar yang paling tinggi menandakan kontributor utama kemenangan alternatif tersebut.")

            with c_deep2:
                # Grafik 2: Matriks Korelasi Cuaca
                corr_matrix = data_historis_full.corr()
                fig_corr = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r", title="2. Matriks Korelasi Antar Teknologi Alamiah", template="plotly_white")
                st.plotly_chart(fig_corr, use_container_width=True)
                st.info("💡 **Makna:** Korelasi negatif (merah) menandakan kedua sumber berlawanan sifat (misal: Air tinggi saat Surya rendah). Ini fundamental untuk *hybrid power balancing*.")

                # Grafik 4: Proyeksi Jejak Karbon Kumulatif
                # Asumsi emisi = Prediksi MW * Emisi per GWh * jam (kasar)
                emisi_proj = {tech: data_ml_tahunan[tech].values * emisi_aktual[i] for i, tech in enumerate(cols_teknologi)}
                fig_emisi = px.area(pd.DataFrame(emisi_proj, index=data_ml_tahunan.index).reset_index().melt(id_vars="Tahun", var_name="Teknologi", value_name="Emisi"), x="Tahun", y="Emisi", color="Teknologi", title="4. Proyeksi Jejak Karbon Sistem", template="plotly_white")
                fig_emisi.update_layout(yaxis_title="Estimasi Emisi (Ton CO2e)")
                st.plotly_chart(fig_emisi, use_container_width=True)
                st.info("💡 **Makna:** Walau terbarukan, beberapa teknologi (seperti Biomassa/Air) tetap menyumbang jejak karbon selama daur hidup (LCA) atau dari metana waduk.")

                # Grafik 6: Sensitivitas Skenario Kebijakan
                skenario_list = ["BAU", "Pajak Karbon", "Subsidi EBT"]
                chart_rows = []
                for sk in skenario_list:
                    em_mult = 1.5 if sk == "Pajak Karbon" else 1.0
                    cap_tmp = capex_awal * ((1 + inflasi) ** selisih_tahun) * (0.7 if sk == "Subsidi EBT" else 1.0)
                    sc = (skor_daya.values / skor_daya.values.max()) * w_tech + (cap_tmp.min() / cap_tmp) * w_econ + (emisi_aktual.min() / (emisi_aktual * em_mult)) * w_env + (sosial_aktual / sosial_aktual.max()) * w_soc
                    for alt, s in zip(cols_teknologi, sc): chart_rows.append({"Skenario": sk, "Teknologi": alt, "Skor": round(float(s), 3)})
                
                fig_sc = px.bar(pd.DataFrame(chart_rows), x="Teknologi", y="Skor", color="Skenario", barmode="group", template="plotly_white", title="6. Sensitivitas Skenario Kebijakan Pemerintah")
                st.plotly_chart(fig_sc, use_container_width=True)
                st.info("💡 **Makna:** Uji kelayakan (robustness). Jika suatu teknologi tetap menang di ketiga skenario (Pajak Karbon vs Subsidi Ekonomi), maka keputusan MCDM sangat kuat secara akademis.")

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan: {e}")
else:
    st.info("👈 Upload file JSON/CSV NASA POWER atau Excel di sidebar untuk memulai.")
