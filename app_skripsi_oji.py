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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Dashboard EBT Kulon Progo - Oji", layout="wide")

st.title("⚡ Dashboard Analisis Potensi EBT & MCDM Kabupaten Kulon Progo")
st.markdown("**Oleh: Muhammad Fachrurrozy (Teknik Fisika UGM)**")
st.divider()

# ==========================================
# SESSION STATE
# ==========================================
if "ai_result_text"  not in st.session_state: st.session_state.ai_result_text  = None
if "ai_result_model" not in st.session_state: st.session_state.ai_result_model = None
if "ai_chart_data"   not in st.session_state: st.session_state.ai_chart_data   = None

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.header("📂 1. Input Data Sistem")
st.sidebar.caption("Support: Excel (.xlsx), NASA POWER Time Series (.csv/.json)")
uploaded_file = st.sidebar.file_uploader("Upload File Data", type=["xlsx", "xls", "csv", "json"])
st.sidebar.divider()

st.sidebar.header("🌍 2. Asumsi Makro & Kebijakan")
inflasi = st.sidebar.number_input("Tingkat Inflasi Tahunan (%)", min_value=0.0, max_value=15.0, value=3.5, step=0.1) / 100
kebijakan = st.sidebar.selectbox(
    "Skenario Kebijakan Transisi",
    ["Business as Usual (BAU)", "Pajak Karbon Tinggi (Pro-Lingkungan)", "Subsidi Masif EBT (Pro-Ekonomi)"]
)
tahun_evaluasi = st.sidebar.slider("Tahun Target Evaluasi MCDM", 2025, 2060, 2040) # Default dipercepat ke 2040 agar lebih relevan
selisih_tahun  = tahun_evaluasi - 2025
st.sidebar.divider()

st.sidebar.header("🧠 3. Integrasi AI")
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    st.sidebar.success("✅ AI terhubung via Server Key.")
except Exception:
    api_key = st.sidebar.text_input("Gemini API Key", type="password", placeholder="Masukkan API Key...", help="Dapatkan di aistudio.google.com")

# ==========================================
# DATA LOKASI EBT (STATIS)
# ==========================================
LOKASI_EBT = [
    {"Teknologi": "PLTMH (Air)", "Kecamatan": "Girimulyo - Perbukitan Menoreh", "Lat": -7.7470, "Lon": 110.1260, "Elevasi": 350, "Potensi_MW": 8.5, "Investasi_M_IDR": 18.5, "Color": "blue", "Icon": "tint", "APBN_Tahap": "RPJMN 2025-2029", "Waktu_Optimal": "2026-2027", "Alasan": "Debit sungai tinggi, infrastruktur jalan sudah ada."},
    {"Teknologi": "PLTMH (Air)", "Kecamatan": "Samigaluh", "Lat": -7.6710, "Lon": 110.1700, "Elevasi": 420, "Potensi_MW": 6.2, "Investasi_M_IDR": 14.0, "Color": "blue", "Icon": "tint", "APBN_Tahap": "RPJMN 2025-2029", "Waktu_Optimal": "2027-2028", "Alasan": "Perlu studi hidrologi lanjutan, akses jalan terbatas."},
    {"Teknologi": "PLTS (Surya)", "Kecamatan": "Wates - Dataran Rendah", "Lat": -7.8600, "Lon": 110.1400, "Elevasi": 15, "Potensi_MW": 22.0, "Investasi_M_IDR": 28.0, "Color": "orange", "Icon": "sun", "APBN_Tahap": "RPJMN 2025-2029", "Waktu_Optimal": "2025-2026", "Alasan": "Irradiasi matahari tinggi, lahan tersedia, dekat jaringan PLN."},
    {"Teknologi": "PLTB (Angin)", "Kecamatan": "Temon / Pantai Glagah", "Lat": -7.9150, "Lon": 110.0760, "Elevasi": 5, "Potensi_MW": 15.0, "Investasi_M_IDR": 35.0, "Color": "green", "Icon": "cloud", "APBN_Tahap": "RPJMN 2030-2034", "Waktu_Optimal": "2030-2032", "Alasan": "Kecepatan angin laut stabil >5 m/s. Butuh kajian lingkungan pesisir."},
    {"Teknologi": "Biomassa", "Kecamatan": "Sentolo / Nanggulan", "Lat": -7.7840, "Lon": 110.2220, "Elevasi": 80, "Potensi_MW": 4.5, "Investasi_M_IDR": 12.0, "Color": "darkred", "Icon": "leaf", "APBN_Tahap": "RPJMN 2025-2029", "Waktu_Optimal": "2028-2029", "Alasan": "Limbah pertanian melimpah, perlu kemitraan petani lokal."},
]
df_lokasi = pd.DataFrame(LOKASI_EBT)

# ==========================================
# FUNGSI SMART PARSER & MACHINE LEARNING (REVISI LOGIKA)
# ==========================================
@st.cache_data
def process_data_and_predict(file_bytes, file_name, tahun_akhir=2060, random_seed=42):
    np.random.seed(random_seed)
    
    # Mapping Parameter NASA ke Teknologi & Faktor Konversi Kasar ke Kapasitas (MW per unit area假设为 1km2 untuk normalisasi)
    # Catatan: Ini adalah pendekatan 'Potential Index' bukan MW absolut kecuali user input luas area
    NASA_MAP = {
        "ALLSKY_SFC_SW_DWN": ("PLTS (Surya)", 0.15), # Irradiasi (MJ/m2) -> Efficiency factor
        "WS10M":             ("PLTB (Angin)", 0.8),  # Wind Speed (m/s) -> Power Coeff
        "PRECTOTCORR":       ("PLTMH (Air)",  0.05), # Precip (mm/day) -> Runoff factor
        "RH2M":              ("Biomassa",     0.1)   # Humidity proxy for biomass growth
    }

    year_data = {}

    # --- PARSING LOGIC ---
    if file_name.endswith('.json'):
        raw_json = json.loads(file_bytes.decode("utf-8"))
        is_nasa_geojson = (
            isinstance(raw_json, dict)
            and raw_json.get("type") == "Feature"
            and "properties" in raw_json
            and "parameter" in raw_json.get("properties", {})
        )

        if is_nasa_geojson:
            fill_val = raw_json.get("header", {}).get("fill_value", -999)
            params   = raw_json["properties"]["parameter"]
            sample_param = list(params.values())[0]
            sample_keys = list(sample_param.keys())
            is_climatology = any(k in ["JAN", "FEB", "ANN"] for k in sample_keys)

            if is_climatology:
                # Climatology: Ambil nilai ANN (Annual Average) sebagai basis tahunan konstan
                # Karena climatology tidak punya tren waktu, kita buat flat line + noise kecil
                for param_key, monthly in params.items():
                    if param_key not in NASA_MAP: continue
                    col_name, factor = NASA_MAP[param_key]
                    
                    val = monthly.get("ANN", fill_val)
                    if val == fill_val:
                        valid_vals = [v for k, v in monthly.items() if k not in ["ANN", "DJF", "MAM", "JJA", "SON"] and v != fill_val]
                        val = np.mean(valid_vals) if valid_vals else 0
                    
                    # Simulasi data 10 tahun terakhir dengan variasi kecil (karena climatology statis)
                    base_val = val * factor
                    for yr in range(2015, 2025):
                        # Tambahkan noise acak kecil agar ML bisa belajar varians
                        noise = np.random.normal(0, base_val * 0.05)
                        year_data.setdefault(yr, {})[col_name] = max(0, base_val + noise)
            
            else:
                # Time Series
                for param_key, monthly in params.items():
                    if param_key not in NASA_MAP: continue
                    col_name, factor = NASA_MAP[param_key]
                    for k, v in monthly.items():
                        if len(str(k)) < 5: continue # Skip invalid keys
                        try:
                            month = int(str(k)[-2:])
                            year = int(str(k)[:4])
                        except:
                            continue
                            
                        if v == fill_val or v is None: continue 
                        
                        # Agregasi bulanan ke tahunan (Average)
                        year_data.setdefault(year, {}).setdefault(col_name, []).append(v * factor)
                
                # Rata-rata per tahun
                for yr in year_data:
                    for col in year_data[yr]:
                        if isinstance(year_data[yr][col], list):
                            year_data[yr][col] = float(np.mean(year_data[yr][col]))

        elif isinstance(raw_json, (list, dict)):
            # Fallback untuk JSON struktur lain
            if isinstance(raw_json, dict):
                df_temp = pd.DataFrame(raw_json)
            else:
                df_temp = pd.DataFrame.from_records(raw_json)
            
            if "Tahun" not in df_temp.columns and "Year" in df_temp.columns:
                df_temp.rename(columns={"Year": "Tahun"}, inplace=True)
            
            if "Tahun" in df_temp.columns:
                df_temp["Tahun"] = pd.to_numeric(df_temp["Tahun"], errors='coerce')
                df_temp = df_temp.dropna(subset=["Tahun"])
                df_temp["Tahun"] = df_temp["Tahun"].astype(int)
                
                numeric_cols = df_temp.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if col != "Tahun":
                        # Asumsi kolom angka adalah potensi langsung jika format custom
                        year_data.setdefault(int(df_temp["Tahun"].iloc[0]), {})[col] = df_temp[col].mean() 
                        # Logika ini sederhana, idealnya loop per baris tapi untuk demo cukup rata-rata
                # Re-parse properly if it looks like time series
                for _, row in df_temp.iterrows():
                    yr = int(row["Tahun"])
                    for c in numeric_cols:
                        if c != "Tahun":
                            year_data.setdefault(yr, {})[c] = row[c]

    elif file_name.endswith('.csv'):
        raw_text = file_bytes.decode("utf-8")
        skip_rows = 0
        for i, line in enumerate(raw_text.split('\n')):
            if "-END HEADER-" in line or ("YEAR" in line and "MO" in line):
                skip_rows = i + 1 if "-END HEADER-" in line else i
                break
        
        try:
            df_raw = pd.read_csv(io.StringIO(raw_text), skiprows=skip_rows)
            df_raw = df_raw.replace(-999.0, np.nan)
            
            if 'YEAR' in df_raw.columns:
                df_clean = pd.DataFrame({'Tahun': df_raw['YEAR'].astype(int)})
                # Konversi ke Indeks Potensi (0-100 scale approximation)
                if 'ALLSKY_SFC_SW_DWN' in df_raw.columns: 
                    df_clean['PLTS (Surya)'] = df_raw['ALLSKY_SFC_SW_DWN'] * 0.15
                if 'WS10M' in df_raw.columns: 
                    df_clean['PLTB (Angin)'] = df_raw['WS10M'] * 0.8
                if 'PRECTOTCORR' in df_raw.columns: 
                    df_clean['PLTMH (Air)'] = df_raw['PRECTOTCORR'] * 0.05
                
                data_input = df_clean.groupby('Tahun').mean()
            else:
                raise ValueError("Format CSV tidak dikenali (Harus ada kolom YEAR).")
        except Exception as e:
            raise ValueError(f"Gagal membaca CSV: {e}")

    else:
        # Excel
        data_input = pd.read_excel(io.BytesIO(file_bytes))
        if "Tahun" in data_input.columns: 
            data_input = data_input.set_index("Tahun")
        else:
            # Asumsi kolom pertama adalah tahun
            data_input.iloc[:, 0] = pd.to_numeric(data_input.iloc[:, 0], errors='coerce')
            data_input = data_input.dropna(subset=[data_input.columns[0]])
            data_input.iloc[:, 0] = data_input.iloc[:, 0].astype(int)
            data_input = data_input.set_index(data_input.columns[0])
            data_input.index.name = "Tahun"

    if not year_data and file_name not in ['.xlsx', '.xls']:
         # Fallback jika parsing JSON/CSV gagal total, gunakan data dummy berdasarkan lokasi
         st.warning("Data file tidak terbaca dengan sempurna. Menggunakan data estimasi berbasis lokasi Kulon Progo.")
         for yr in range(2015, 2025):
             year_data[yr] = {
                 "PLTS (Surya)": 4.5 + np.random.normal(0, 0.2),
                 "PLTB (Angin)": 3.2 + np.random.normal(0, 0.3),
                 "PLTMH (Air)": 2.1 + np.random.normal(0, 0.1)
             }
        data_input = pd.DataFrame.from_dict(year_data, orient='index')

    if year_data and file_name not in ['.xlsx', '.xls']:
        data_input = pd.DataFrame.from_dict(year_data, orient='index')
        data_input.index.name = "Tahun"

    # --- MACHINE LEARNING REVISION ---
    # Masalah sebelumnya: Regresi pada data mentah yang fluktuatif tanpa tren jelas.
    # Solusi: Gunakan Moving Average untuk smoothing sebelum regresi, atau batasi degree polinomial.
    
    tahun_prediksi = np.arange(2025, tahun_akhir + 1)
    X_pred  = tahun_prediksi.reshape(-1, 1).astype(float)
    
    # Gabungkan data historis dan tahun prediksi untuk index
    all_years = np.concatenate([data_input.index.values, tahun_prediksi])
    
    data_ml_tahunan = pd.DataFrame(index=pd.Index(tahun_prediksi, name='Tahun'))
    range_bulan = pd.date_range(start="2025-01-01", end=f"{tahun_akhir}-12-31", freq="MS")
    data_ml_bulan = pd.DataFrame(index=range_bulan)
    
    metrics = {}
    rng = np.random.default_rng(seed=random_seed)

    for col in data_input.columns:
        y_train = data_input[col].values.astype(float)
        X_train = data_input.index.values.reshape(-1, 1).astype(float)
        
        # Smoothing data historis agar tren lebih terlihat (Optional, tapi membantu stabilitas)
        # Kita biarkan asli tapi gunakan Degree 1 (Linear) jika data sedikit, Degree 2 jika cukup
        degree = 2 if len(y_train) > 5 else 1
        
        pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('reg', LinearRegression())
        ])
        
        try:
            pipeline.fit(X_train, y_train)
            y_pred_hist = pipeline.predict(X_train)
            r2 = r2_score(y_train, y_pred_hist)
            
            # Jika R2 sangat buruk (<0), artinya data sangat acak. Paksa model linear sederhana atau flat.
            if r2 < 0:
                pipeline = Pipeline([
                    ('poly', PolynomialFeatures(degree=1, include_bias=False)),
                    ('reg', LinearRegression())
                ])
                pipeline.fit(X_train, y_train)
                y_pred_hist = pipeline.predict(X_train)
                r2 = r2_score(y_train, y_pred_hist)
                
            metrics[col] = {"R²": round(r2, 3), "MAE": round(mean_absolute_error(y_train, y_pred_hist), 3)}
        except Exception as e:
            metrics[col] = {"R²": 0.0, "MAE": 0.0}
            # Fallback flat prediction
            pipeline = None 

        # Prediksi Tahunan
        if pipeline:
            tren_tahunan = pipeline.predict(X_pred)
        else:
            tren_tahunan = np.full(len(tahun_prediksi), np.mean(y_train))
            
        # Tambahkan Noise Realistis (tidak boleh negatif)
        noise_std = np.std(y_train) * 0.5 if np.std(y_train) > 0 else 0.1
        noise_tahun = rng.normal(0, noise_std, len(tahun_prediksi))
        final_pred = np.maximum(0, tren_tahunan + noise_tahun)
        
        # LOGIC FIX: Pastikan prediksi tidak meledak (caping at 2x max historical unless strong trend)
        max_hist = np.max(y_train) * 1.5
        final_pred = np.minimum(final_pred, max_hist) 
        
        data_ml_tahunan[col] = final_pred

        # Prediksi Bulanan (Seasonality)
        tahun_bulan = range_bulan.year.values.astype(float).reshape(-1, 1)
        if pipeline:
            tren_bulan = pipeline.predict(tahun_bulan)
        else:
            tren_bulan = np.full(len(range_bulan), np.mean(y_train))
            
        base_val = float(np.mean(y_train))
        # Sinusoidal seasonality
        seasonality = 0.15 * base_val * np.sin(2 * np.pi * range_bulan.month / 12)
        noise_bulan = rng.normal(0, base_val * 0.05, len(range_bulan))
        data_ml_bulan[col] = np.maximum(0, tren_bulan + seasonality + noise_bulan)

    return data_input, data_ml_tahunan, data_ml_bulan, metrics

def get_prediction_for_year(data_ml: pd.DataFrame, tahun: int) -> pd.Series:
    if tahun in data_ml.index: 
        return data_ml.loc[tahun]
    # Interpolasi jika tahun tidak ada
    return data_ml.reindex([tahun], method='nearest').iloc[0]


# ==========================================
# MAIN CONTENT
# ==========================================
if uploaded_file is not None:
    try:
        file_bytes = uploaded_file.getvalue()
        file_name  = uploaded_file.name

        data_historis, data_ml_tahunan, data_ml_bulan, metrics = process_data_and_predict(file_bytes, file_name, tahun_akhir=tahun_evaluasi)

        # Ringkasan 5 Tahun
        bins = list(range(2025, tahun_evaluasi + 6, 5))
        if tahun_evaluasi % 5 != 0: bins.append(tahun_evaluasi + 1)
        labels_5 = [f"{y}–{min(y+4, tahun_evaluasi)}" for y in bins[:-1]]
        
        ml_copy = data_ml_tahunan.copy()
        ml_copy['Periode'] = pd.cut(ml_copy.index, bins=bins, labels=labels_5, right=False)
        data_5_tahun = ml_copy.groupby('Periode', observed=True).mean()

        # Parameter MCDM Dasar (Disesuaikan dengan jumlah kolom data)
        n_alt = len(data_historis.columns)
        # Default values jika data tidak spesifik menyebutkan biaya/emisi per teknologi
        base_capex  = [12.0, 18.0, 22.0, 25.0, 30.0][:n_alt] 
        base_emisi  = [40.0, 11.0, 24.0, 230.0, 60.0][:n_alt]
        base_sosial = [90, 70, 85, 80, 75][:n_alt]
        
        capex_awal    = np.array(base_capex)
        emisi_aktual  = np.array(base_emisi)
        sosial_aktual = np.array(base_sosial)

        tab1, tab2, tab3 = st.tabs(["📈 Analisis Prediksi (ML)", "⚖️ MCDM & Kebijakan", "🗺️ Peta Potensi Spasial"])

        # ---------------- TAB 1: PREDIKSI ML ----------------
        with tab1:
            st.header(f"Proyeksi Potensi Daya hingga {tahun_evaluasi}")
            
            st.subheader("📊 Evaluasi Kualitas Model")
            metrics_df = pd.DataFrame(metrics).T
            st.dataframe(metrics_df.style.format("{:.3f}").background_gradient(subset=["R²"], cmap="RdYlGn"), use_container_width=True)
            st.caption("Catatan: R² rendah wajar untuk data iklim karena sifatnya yang stokastik (acak), bukan tren deterministik.")
            st.divider()

            # FIX VISUALISASI: Pisahkan Historis dan Prediksi atau gunakan Dual Axis jika skala beda jauh
            # Di sini kita gunakan Plotly Area Chart transparan agar tumpang tindih terlihat logis
            st.write("**Tren Potensi Energi (Histori vs Proyeksi):**")
            
            df_hist = data_historis.copy().reset_index()
            df_hist['Tipe'] = 'Data Historis (Input)'
            df_pred = data_ml_tahunan.copy().reset_index()
            df_pred['Tipe'] = 'Prediksi ML (Output)'

            # Melt data
            cols_bersama = [c for c in df_hist.columns if c in df_pred.columns and c not in ['Tipe']]
            
            fig_tren = go.Figure()
            
            # Plot Historis
            for col in cols_bersama:
                if col == 'Tahun': continue
                fig_tren.add_trace(go.Scatter(
                    x=df_hist['Tahun'], y=df_hist[col],
                    mode='lines+markers', name=f'{col} (Hist)',
                    line=dict(width=2, dash='dot'), opacity=0.6
                ))
            
            # Plot Prediksi
            for col in cols_bersama:
                if col == 'Tahun': continue
                fig_tren.add_trace(go.Scatter(
                    x=df_pred['Tahun'], y=df_pred[col],
                    mode='lines', name=f'{col} (Prediksi)',
                    line=dict(width=3), fill='tozeroy', fillcolor='rgba(0,100,200,0.1)'
                ))

            fig_tren.update_layout(
                template='plotly_dark',
                xaxis_title="Tahun",
                yaxis_title="Indeks Potensi / Kapasitas (Norm)",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_tren, use_container_width=True)

            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.write("**Rata-rata Potensi Per 5 Tahun:**")
                fig_bar = px.bar(
                    data_5_tahun.reset_index().melt(id_vars="Periode", var_name="Teknologi", value_name="Nilai"),
                    x="Periode", y="Nilai", color="Teknologi", barmode="group",
                    template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Set2
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with col_c2:
                st.write(f"**Variabilitas Bulanan ({tahun_evaluasi}):**")
                data_tahun_terakhir = data_ml_bulan[data_ml_bulan.index.year == tahun_evaluasi].copy()
                if not data_tahun_terakhir.empty:
                    data_tahun_terakhir['Bulan'] = data_tahun_terakhir.index.month_name()
                    fig_line = px.line(
                        data_tahun_terakhir.reset_index().melt(id_vars="Bulan", var_name="Teknologi", value_name="Nilai"),
                        x="Bulan", y="Nilai", color="Teknologi", markers=True,
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig_line, use_container_width=True)

        # ---------------- TAB 2: MCDM ----------------
        with tab2:
            st.header("MCDM dengan Parameter Dinamis")
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
                st.write("### Bobot Kriteria")
                w_tech = st.slider("Potensi Daya (Benefit)", 0.0, 1.0, 0.4, key="w_tech")
                w_econ = st.slider("Biaya Investasi (Cost)", 0.0, 1.0, 0.3, key="w_econ")
                w_env  = st.slider("Emisi Karbon (Cost)", 0.0, 1.0, 0.2, key="w_env") * pengali_emisi
                w_soc  = st.slider("Penerimaan Sosial (Benefit)", 0.0, 1.0, 0.1, key="w_soc")
                total  = w_tech + w_econ + w_env + w_soc
                if total == 0: total = 0.001 # Avoid division by zero
                w_tech, w_econ, w_env, w_soc = w_tech/total, w_econ/total, w_env/total, w_soc/total

            with col_b2:
                skor_daya = get_prediction_for_year(data_ml_tahunan, tahun_evaluasi)
                data_aktual = pd.DataFrame({
                    "Daya Prediksi (MW)": skor_daya.values,
                    "Investasi Terinflasi (M IDR/MW)": capex_terinflasi,
                    "Emisi (Ton CO2e/GWh)": emisi_aktual,
                    "Sosial (1-100)": sosial_aktual,
                }, index=data_historis.columns)
                st.dataframe(data_aktual.style.format("{:.2f}").background_gradient(cmap="Blues"), use_container_width=True)

            # Normalisasi MCDM (MinMax)
            norm_df = pd.DataFrame(index=data_aktual.index)
            # Benefit Criteria (Makin besar makin baik)
            norm_df["Daya"]      = data_aktual["Daya Prediksi (MW)"] / data_aktual["Daya Prediksi (MW)"].max()
            norm_df["Sosial"]    = data_aktual["Sosial (1-100)"] / data_aktual["Sosial (1-100)"].max()
            # Cost Criteria (Makin kecil makin baik) -> Invers
            min_inv = data_aktual["Investasi Terinflasi (M IDR/MW)"].min()
            norm_df["Investasi"] = min_inv / data_aktual["Investasi Terinflasi (M IDR/MW)"]
            
            min_em = data_aktual["Emisi (Ton CO2e/GWh)"].min()
            norm_df["Emisi"]     = min_em / data_aktual["Emisi (Ton CO2e/GWh)"]

            # Hitung Skor Akhir
            skor_akhir = (norm_df["Daya"]*w_tech + norm_df["Investasi"]*w_econ + norm_df["Emisi"]*w_env + norm_df["Sosial"]*w_soc)
            hasil_df = pd.DataFrame(skor_akhir, columns=["Skor Preferensi"]).sort_values("Skor Preferensi", ascending=False)

            st.divider()
            st.subheader(f"🏆 Rekomendasi Prioritas Tahun {tahun_evaluasi}")
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                fig_rank = px.bar(
                    hasil_df.reset_index().rename(columns={"index": "Teknologi"}),
                    x="Skor Preferensi", y="Teknologi", orientation="h", 
                    color="Skor Preferensi", color_continuous_scale="Viridis",
                    template="plotly_dark", text_auto='.3f'
                )
                fig_rank.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig_rank, use_container_width=True)

            with col_r2:
                pemenang = hasil_df.index[0]
                skor_pemenang = hasil_df.iloc[0]['Skor Preferensi']
                st.success(f"**Pemenang:** {pemenang} | Skor: {skor_pemenang:.3f}")
                
                bobot_label = {"Daya": w_tech, "Investasi": w_econ, "Emisi": w_env, "Sosial": w_soc}
                categories = list(bobot_label.keys())
                weighted_df = pd.DataFrame({dim: norm_df[dim] * bobot_label[dim] for dim in categories}, index=norm_df.index)

                fig_radar = go.Figure()
                for alt in weighted_df.index:
                    vals = weighted_df.loc[alt, categories].tolist()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=vals + [vals[0]],
                        theta=[f"{d}" for d in categories] + [categories[0]],
                        fill='toself', name=alt,
                        hovertemplate=f"{alt}<br>%{{theta}}: %{{r:.2f}}<extra></extra>"
                    ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, max(w_tech, w_econ, w_env, w_soc) * 1.1])),
                    template="plotly_dark", height=350, showlegend=True
                )
                st.plotly_chart(fig_radar, use_container_width=True)

            # AI INSIGHT
            st.divider()
            st.subheader("🤖 AI Executive Summary")
            col_ai1, col_ai2 = st.columns([1, 3])
            with col_ai1: generate_btn = st.button("✨ Generate AI Insight")
            with col_ai2:
                if st.session_state.ai_result_model: st.caption(f"Model: `{st.session_state.ai_result_model}`")

            if generate_btn:
                if not api_key: st.warning("⚠️ Masukkan API Key di sidebar.")
                else:
                    try:
                        client = genai.Client(api_key=api_key)
                        prompt_ai = f"""Anda adalah pakar transisi energi. Analisis data berikut untuk Kulon Progo tahun {tahun_evaluasi}:
                        1. Skenario: {kebijakan}
                        2. Inflasi: {inflasi*100:.1f}%
                        3. Ranking Akhir: {hasil_df.head(2).to_string()}
                        4. Nilai Ternormalisasi: {norm_df.to_string()}
                        
                        Berikan output HANYA dalam format JSON murni (tanpa markdown code block):
                        {{
                            "mengapa_menang": "Alasan teknis singkat mengapa teknologi ini menang.",
                            "dampak_ekonomi": "Analisis dampak biaya/investasi.",
                            "rekomendasi": "Saran strategis 1 kalimat."
                        }}
                        """
                        with st.spinner("AI sedang menganalisis..."):
                            response = client.models.generate_content(model="gemini-2.0-flash-exp", contents=prompt_ai)
                            raw = response.text.strip()
                            # Clean markdown if present
                            if raw.startswith("```"): raw = raw.replace("```json", "").replace("```", "")
                            st.session_state.ai_result_text = json.loads(raw)
                            st.session_state.ai_result_model = "gemini-2.0-flash-exp"
                    except Exception as e: 
                        st.error(f"Kesalahan AI: {e}")
                        st.session_state.ai_result_text = {"mengapa_menang": "Error analisis", "dampak_ekonomi": "-", "rekomendasi": "-"}

            if st.session_state.ai_result_text:
                res = st.session_state.ai_result_text
                c1, c2, c3 = st.columns(3)
                c1.info(f"🏆 **Mengapa Menang?**\n\n{res.get('mengapa_menang','')}")
                c2.warning(f"💰 **Dampak Ekonomi**\n\n{res.get('dampak_ekonomi','')}")
                c3.success(f"📋 **Rekomendasi**\n\n{res.get('rekomendasi','')}")

        # ---------------- TAB 3: PETA 3D ----------------
        with tab3:
            st.header("🗺️ Pemetaan Geospasial Potensi EBT Kulon Progo")
            mode_peta = st.radio("Mode Tampilan Peta:", ["🗺️ 2D Interaktif", "🏔️ 3D Globe (Elevasi)"], horizontal=True)

            if mode_peta == "🗺️ 2D Interaktif":
                m = folium.Map(location=[-7.8288, 110.1587], zoom_start=11, tiles="CartoDB dark_matter")
                for _, row in df_lokasi.iterrows():
                    popup_html = f"""
                    <div style="font-family: sans-serif;">
                        <b>{row['Teknologi']}</b><br>
                        📍 {row['Kecamatan']}<br>
                        ⚡ {row['Potensi_MW']} MW<br>
                        💰 Rp {row['Investasi_M_IDR']} M
                    </div>
                    """
                    folium.Marker(
                        [row['Lat'], row['Lon']], 
                        popup=folium.Popup(popup_html, max_width=260), 
                        icon=folium.Icon(color=row['Color'], icon=row['Icon'], prefix='fa')
                    ).add_to(m)
                st_folium(m, width=None, height=500, use_container_width=True)
            else:
                color_map = {"PLTMH (Air)": "#29b6f6", "PLTS (Surya)": "#ffa726", "PLTB (Angin)": "#66bb6a", "Biomassa": "#ef5350"}
                fig_3d = go.Figure()
                for _, row in df_lokasi.iterrows():
                    fig_3d.add_trace(go.Scatter3d(
                        x=[row['Lon']], y=[row['Lat']], z=[row['Elevasi']], 
                        mode='markers+text',
                        marker=dict(size=row['Potensi_MW'] * 1.5, color=color_map.get(row['Teknologi'], "#fff"), opacity=0.85, line=dict(width=0.5, color='white')),
                        text=[f"<b>{row['Teknologi']}</b><br>{row['Potensi_MW']} MW"], 
                        textposition="top center",
                        name=row['Teknologi']
                    ))
                fig_3d.update_layout(
                    scene=dict(
                        bgcolor="rgb(10,15,30)", 
                        xaxis=dict(title="Longitude", gridcolor="#1e3a5f"), 
                        yaxis=dict(title="Latitude", gridcolor="#1e3a5f"), 
                        zaxis=dict(title="Elevasi (m)", gridcolor="#1e3a5f")
                    ), 
                    template="plotly_dark", 
                    height=600,
                    margin=dict(l=0, r=0, b=0, t=0)
                )
                st.plotly_chart(fig_3d, use_container_width=True)

            st.divider()
            st.subheader("📋 Detail Potensi, Investasi & Jadwal APBN")
            st.dataframe(df_lokasi[["Teknologi", "Kecamatan", "Potensi_MW", "Investasi_M_IDR", "Waktu_Optimal", "APBN_Tahap"]].style.background_gradient(subset=["Potensi_MW"], cmap="Blues"), use_container_width=True)

            st.subheader("🗓️ Timeline Optimal Pembangunan vs Siklus APBN")
            timeline_data = []
            for _, r in df_lokasi.iterrows():
                start_y = int(r["Waktu_Optimal"].split("-")[0])
                end_y = int(r["Waktu_Optimal"].split("-")[1])
                timeline_data.append({
                    "Teknologi": r['Teknologi'], 
                    "Mulai": pd.Timestamp(f"{start_y}-01-01"), 
                    "Selesai": pd.Timestamp(f"{end_y}-12-31"), 
                    "APBN": r["APBN_Tahap"], 
                    "MW": r["Potensi_MW"]
                })
            df_tl = pd.DataFrame(timeline_data)
            
            fig_tl = px.timeline(
                df_tl, 
                x_start="Mulai", x_end="Selesai", y="Teknologi", 
                color="APBN", hover_data=["MW"],
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Prism
            )
            fig_tl.update_yaxes(autorange="reversed", title="Teknologi")
            fig_tl.update_xaxis(title="Waktu Implementasi")
            st.plotly_chart(fig_tl, use_container_width=True)

    except Exception as e:
        st.error(f"Terjadi kesalahan sistem: {e}")
        st.exception(e)

else:
    st.info("👈 **Silakan Upload File Data** di sidebar kiri untuk memulai analisis.\n\nFormat yang didukung:\n- **NASA POWER CSV/JSON** (Time Series atau Climatology)\n- **Excel Custom** (Kolom: Tahun, PLTS, PLTB, dll)")
    
    # Demo Data Generator Button
    if st.button("Generate Contoh Data Dummy (Untuk Testing)"):
        import csv
        import io
        output = io.StringIO()
        fieldnames = ['YEAR', 'MO', 'ALLSKY_SFC_SW_DWN', 'WS10M', 'PRECTOTCORR']
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for y in range(2015, 2025):
            for m in range(1, 13):
                writer.writerow({
                    'YEAR': y,
                    'MO': m,
                    'ALLSKY_SFC_SW_DWN': 15 + np.random.normal(0, 2), # Solar
                    'WS10M': 4.5 + np.random.normal(0, 0.5), # Wind
                    'PRECTOTCORR': 5 + np.random.normal(0, 1) # Rain
                })
        st.download_button(label="Download Contoh CSV", data=output.getvalue(), file_name="dummy_nasa_kulonprogo.csv", mime="text/csv")
