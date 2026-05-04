import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from google import genai
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
# SESSION STATE — AI summary tidak hilang saat pindah tab
# ==========================================
if "ai_result_text"  not in st.session_state: st.session_state.ai_result_text  = None
if "ai_result_model" not in st.session_state: st.session_state.ai_result_model = None
if "ai_chart_data"   not in st.session_state: st.session_state.ai_chart_data   = None

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.header("📂 1. Input Data Sistem")
st.sidebar.caption("Support: Excel (.xlsx), NASA POWER (.csv), atau JSON (.json)")
uploaded_file = st.sidebar.file_uploader("Upload File Data", type=["xlsx", "xls", "csv", "json"])
st.sidebar.divider()

st.sidebar.header("🌍 2. Asumsi Makro & Kebijakan")
inflasi = st.sidebar.number_input(
    "Tingkat Inflasi Tahunan (%)", min_value=0.0, max_value=15.0, value=3.5, step=0.1
) / 100
kebijakan = st.sidebar.selectbox(
    "Skenario Kebijakan Transisi",
    ["Business as Usual (BAU)", "Pajak Karbon Tinggi (Pro-Lingkungan)", "Subsidi Masif EBT (Pro-Ekonomi)"]
)
tahun_evaluasi = st.sidebar.slider("Tahun Target Evaluasi MCDM", 2025, 2060, 2060)
selisih_tahun  = tahun_evaluasi - 2025
st.sidebar.divider()

st.sidebar.header("🧠 3. Integrasi AI")
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    st.sidebar.success("✅ AI terhubung via Server Key.")
except Exception:
    api_key = st.sidebar.text_input(
        "Gemini API Key", type="password",
        placeholder="Masukkan API Key...",
        help="Dapatkan di aistudio.google.com"
    )

# ==========================================
# DATA LOKASI EBT
# ==========================================
LOKASI_EBT = [
    {
        "Teknologi": "PLTMH (Air)", "Kecamatan": "Girimulyo - Perbukitan Menoreh",
        "Lat": -7.7470, "Lon": 110.1260, "Elevasi": 350,
        "Potensi_MW": 8.5, "Investasi_M_IDR": 18.5, "Color": "blue", "Icon": "tint",
        "APBN_Tahap": "RPJMN 2025-2029", "Waktu_Optimal": "2026-2027",
        "Alasan": "Debit sungai tinggi, infrastruktur jalan sudah ada."
    },
    {
        "Teknologi": "PLTMH (Air)", "Kecamatan": "Samigaluh",
        "Lat": -7.6710, "Lon": 110.1700, "Elevasi": 420,
        "Potensi_MW": 6.2, "Investasi_M_IDR": 14.0, "Color": "blue", "Icon": "tint",
        "APBN_Tahap": "RPJMN 2025-2029", "Waktu_Optimal": "2027-2028",
        "Alasan": "Perlu studi hidrologi lanjutan, akses jalan terbatas."
    },
    {
        "Teknologi": "PLTS (Surya)", "Kecamatan": "Wates - Dataran Rendah",
        "Lat": -7.8600, "Lon": 110.1400, "Elevasi": 15,
        "Potensi_MW": 22.0, "Investasi_M_IDR": 28.0, "Color": "orange", "Icon": "sun",
        "APBN_Tahap": "RPJMN 2025-2029", "Waktu_Optimal": "2025-2026",
        "Alasan": "Irradiasi matahari tinggi, lahan tersedia, dekat jaringan PLN."
    },
    {
        "Teknologi": "PLTB (Angin)", "Kecamatan": "Temon / Pantai Glagah",
        "Lat": -7.9150, "Lon": 110.0760, "Elevasi": 5,
        "Potensi_MW": 15.0, "Investasi_M_IDR": 35.0, "Color": "green", "Icon": "cloud",
        "APBN_Tahap": "RPJMN 2030-2034", "Waktu_Optimal": "2030-2032",
        "Alasan": "Kecepatan angin laut stabil >5 m/s. Butuh kajian lingkungan pesisir."
    },
    {
        "Teknologi": "Biomassa", "Kecamatan": "Sentolo / Nanggulan",
        "Lat": -7.7840, "Lon": 110.2220, "Elevasi": 80,
        "Potensi_MW": 4.5, "Investasi_M_IDR": 12.0, "Color": "darkred", "Icon": "leaf",
        "APBN_Tahap": "RPJMN 2025-2029", "Waktu_Optimal": "2028-2029",
        "Alasan": "Limbah pertanian melimpah, perlu kemitraan petani lokal."
    },
]
df_lokasi = pd.DataFrame(LOKASI_EBT)


# ==========================================
# FUNGSI SMART PARSER & MACHINE LEARNING
# ==========================================
@st.cache_data
def process_data_and_predict(file_bytes, file_name, tahun_akhir=2060, random_seed=42):
    """
    Membaca file (NASA CSV atau Excel), melatih Polynomial Regression per kolom,
    menghasilkan prediksi tahunan + bulanan, dan mengembalikan metrik evaluasi.

    Return:
        data_historis   : DataFrame historis (index = Tahun int)
        data_ml_tahunan : DataFrame prediksi tahunan 2025–tahun_akhir
        data_ml_bulan   : DataFrame prediksi bulanan 2025–tahun_akhir (dengan seasonality)
        metrics         : dict {kolom: {R², MAE}}
    """
    np.random.seed(random_seed)

    # ── SMART PARSER ──────────────────────────────────────────────────────────
    if file_name.endswith('.json'):
        """
        Format JSON yang didukung (dua varian):

        Varian 1 — Array of objects (satu baris = satu tahun):
        [
          {"Tahun": 2020, "PLTS (Surya)": 12.5, "PLTB (Angin)": 8.2},
          {"Tahun": 2021, "PLTS (Surya)": 13.0, "PLTB (Angin)": 8.5}
        ]

        Varian 2 — Object of arrays (kolom-kolom sebagai key):
        {
          "Tahun":        [2020, 2021, 2022],
          "PLTS (Surya)": [12.5, 13.0, 14.1],
          "PLTB (Angin)": [8.2,  8.5,  8.1]
        }
        """
        raw_json = json.loads(file_bytes.decode("utf-8"))

        # Normalise kedua varian ke DataFrame
        if isinstance(raw_json, list):
            # Varian 1: list of dicts  → [{Tahun:2020, PLTS:12.5}, ...]
            df_json = pd.DataFrame.from_records(raw_json)
        elif isinstance(raw_json, dict):
            # Varian 2: dict of lists  → {Tahun:[2020,...], PLTS:[12.5,...]}
            # Konversi setiap value ke pd.Series agar pandas tidak ambiguous
            df_json = pd.DataFrame({k: pd.Series(v) for k, v in raw_json.items()})
        else:
            raise ValueError(
                "Format JSON tidak dikenali. Gunakan array of objects "
                "[{\"Tahun\":2020, ...}] atau object of arrays "
                "{\"Tahun\":[2020,...], ...}."
            )

        # Cari kolom tahun (case-insensitive)
        tahun_col = next(
            (c for c in df_json.columns if c.strip().lower() == "tahun"), None
        )
        if tahun_col is None:
            raise ValueError(
                "File JSON harus memiliki kolom 'Tahun' "
                "(sebagai kunci tahun data historis)."
            )

        df_json = df_json.rename(columns={tahun_col: "Tahun"})
        df_json["Tahun"] = df_json["Tahun"].astype(int)

        # Kolom numerik selain Tahun = kolom teknologi EBT
        kolom_ebt = [c for c in df_json.columns if c != "Tahun"]
        if not kolom_ebt:
            raise ValueError("File JSON tidak memiliki kolom data EBT selain 'Tahun'.")

        # Pastikan semua kolom EBT numerik
        for c in kolom_ebt:
            df_json[c] = pd.to_numeric(df_json[c], errors='coerce')

        data_input = df_json.set_index("Tahun")[kolom_ebt].dropna(how="all")

    elif file_name.endswith('.csv'):
        raw_text = file_bytes.decode("utf-8")
        lines    = raw_text.split('\n')
        skip_rows = 0

        for i, line in enumerate(lines):
            if "-END HEADER-" in line:
                skip_rows = i + 1
                break
            elif "YEAR" in line and "MO" in line:
                skip_rows = i
                break

        df_raw = pd.read_csv(io.StringIO(raw_text), skiprows=skip_rows)

        if 'YEAR' not in df_raw.columns:
            raise ValueError(
                "File CSV yang diunggah bukan data historis cuaca. "
                "Pastikan mengunduh 'Time Series' dari NASA POWER "
                "yang memiliki kolom YEAR, MO, DY."
            )

        df_raw = df_raw.replace(-999.0, np.nan).ffill().bfill()

        df_clean = pd.DataFrame()
        df_clean['Tahun'] = df_raw['YEAR'].astype(int)

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
            raise ValueError(
                "File CSV tidak mengandung parameter EBT yang dibutuhkan "
                "(ALLSKY_SFC_SW_DWN, WS10M, atau PRECTOTCORR)."
            )

        data_input = df_clean.groupby('Tahun').mean()

    else:
        data_input = pd.read_excel(io.BytesIO(file_bytes))
        if "Tahun" in data_input.columns:
            data_input = data_input.set_index("Tahun")

    # Pastikan index bertipe int
    data_input.index = data_input.index.astype(int)

    # ── MACHINE LEARNING: Polynomial Regression ────────────────────────────────
    tahun_prediksi = np.arange(2025, tahun_akhir + 1)
    X_train = data_input.index.values.reshape(-1, 1).astype(float)
    X_pred  = tahun_prediksi.reshape(-1, 1).astype(float)

    # DataFrame prediksi TAHUNAN
    data_ml_tahunan = pd.DataFrame(index=pd.Index(tahun_prediksi, name='Tahun'))

    # DataFrame prediksi BULANAN (dengan seasonality, seperti versi Oji)
    range_bulan   = pd.date_range(start="2025-01-01", end=f"{tahun_akhir}-12-31", freq="MS")
    data_ml_bulan = pd.DataFrame(index=range_bulan)

    metrics = {}
    rng = np.random.default_rng(seed=random_seed)

    for col in data_input.columns:
        y_train   = data_input[col].values.astype(float)
        base_val  = float(np.mean(y_train))
        noise_std = np.std(y_train) * 0.3   # noise lebih konservatif

        # ── 1. Polynomial Regression degree=2 ──
        pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('reg',  LinearRegression())
        ])
        pipeline.fit(X_train, y_train)

        # Evaluasi pada data historis
        y_pred_hist = pipeline.predict(X_train)
        metrics[col] = {
            "R²":  round(r2_score(y_train, y_pred_hist), 3),
            "MAE": round(mean_absolute_error(y_train, y_pred_hist), 3)
        }

        # ── 2. Prediksi tahunan (tren ML + noise kecil) ──
        tren_tahunan = pipeline.predict(X_pred)
        noise_tahun  = rng.normal(0, noise_std, len(tahun_prediksi))
        data_ml_tahunan[col] = np.maximum(0, tren_tahunan + noise_tahun)

        # ── 3. Prediksi bulanan (tren ML per tahun + seasonality + noise bulanan) ──
        tahun_bulan = range_bulan.year.values.astype(float).reshape(-1, 1)
        tren_bulan  = pipeline.predict(tahun_bulan)
        seasonality = 0.2 * base_val * np.sin(2 * np.pi * range_bulan.month / 12)
        noise_bulan = rng.normal(0, base_val * 0.05, len(range_bulan))
        data_ml_bulan[col] = np.maximum(0, tren_bulan + seasonality + noise_bulan)

    return data_input, data_ml_tahunan, data_ml_bulan, metrics


def get_prediction_for_year(data_ml: pd.DataFrame, tahun: int) -> pd.Series:
    """Ambil prediksi untuk tahun tertentu dengan fallback ke tahun terdekat."""
    if tahun in data_ml.index:
        return data_ml.loc[tahun]
    closest = data_ml.index[np.argmin(np.abs(data_ml.index - tahun))]
    st.warning(f"Tahun {tahun} tidak ditemukan, menggunakan tahun terdekat: {closest}.")
    return data_ml.loc[closest]


# ==========================================
# MAIN CONTENT
# ==========================================
if uploaded_file is not None:
    try:
        file_bytes = uploaded_file.getvalue()
        file_name  = uploaded_file.name

        data_historis, data_ml_tahunan, data_ml_bulan, metrics = process_data_and_predict(
            file_bytes, file_name, tahun_akhir=2060
        )

        # Ringkasan per 5 tahun menggunakan pd.cut (lebih akurat)
        bins      = list(range(2025, 2066, 5))
        labels_5  = [f"{y}–{y+4}" for y in bins[:-1]]
        ml_copy   = data_ml_tahunan.copy()
        ml_copy['Periode'] = pd.cut(ml_copy.index, bins=bins, labels=labels_5, right=False)
        data_5_tahun = ml_copy.groupby('Periode', observed=True).mean()

        # Parameter MCDM — aman untuk n_alt hingga 5
        n_alt         = len(data_historis.columns)
        base_capex    = [12.0, 18.0, 22.0, 25.0, 30.0]
        base_emisi    = [40.0, 11.0, 24.0, 230.0, 60.0]
        base_sosial   = [90,   70,   85,   80,    75  ]
        capex_awal    = np.array(base_capex[:n_alt])
        emisi_aktual  = np.array(base_emisi[:n_alt])
        sosial_aktual = np.array(base_sosial[:n_alt])

        # ====================================================
        # TABS
        # ====================================================
        tab1, tab2, tab3 = st.tabs([
            "📈 Analisis Prediksi (ML)",
            "⚖️ MCDM & Kebijakan",
            "🗺️ Peta Potensi Spasial"
        ])

        # ====================================================
        # TAB 1: PREDIKSI ML
        # ====================================================
        with tab1:
            st.header(f"Proyeksi Potensi Daya hingga {tahun_evaluasi}")

            if file_name.endswith('.csv'):
                st.success(
                    "🛰️ **NASA POWER Data Detected!** Metadata otomatis dipotong, "
                    "*missing values* (-999.0) diatasi, dan parameter iklim dinormalisasi "
                    "menjadi ekuivalen kapasitas daya."
                )
            elif file_name.endswith('.json'):
                st.success(
                    "📋 **JSON Data Detected!** Mendukung dua varian: "
                    "*array of objects* `[{\"Tahun\":2020, ...}]` maupun "
                    "*object of arrays* `{\"Tahun\":[2020,...], ...}`."
                )

            # ── Metrik Evaluasi Model ──────────────────────────────────────────
            st.subheader("📊 Evaluasi Kualitas Model (Polynomial Regression Degree-2)")
            metrics_df = pd.DataFrame(metrics).T
            col_m1, col_m2 = st.columns([1, 2])
            with col_m1:
                st.dataframe(
                    metrics_df.style
                    .format("{:.3f}")
                    .background_gradient(subset=["R²"],  cmap="RdYlGn")
                    .background_gradient(subset=["MAE"], cmap="RdYlGn_r"),
                    use_container_width=True
                )
            with col_m2:
                st.info(
                    "**R²** mendekati 1.0 → model menjelaskan variansi data dengan baik. "
                    "**MAE** (Mean Absolute Error) = rata-rata kesalahan prediksi dalam satuan MW. "
                    "Model menggunakan Polynomial Regression degree-2 untuk menangkap tren non-linear."
                )

            st.divider()

            # ── Grafik Historis vs Prediksi (tren tahunan) ────────────────────
            st.write("**Histori Data vs Tren Prediksi ML (Tahunan, MW):**")

            df_hist = data_historis.copy()
            df_hist.index.name = 'Tahun'
            df_hist = df_hist.reset_index()
            df_hist['Tipe'] = 'Historis'

            df_pred = data_ml_tahunan.copy().reset_index()
            df_pred['Tipe'] = 'Prediksi'

            cols_bersama = [c for c in df_hist.columns if c in df_pred.columns]
            df_gabung = pd.concat([df_hist[cols_bersama], df_pred[cols_bersama]], ignore_index=True)
            df_melt   = df_gabung.melt(id_vars=['Tahun', 'Tipe'], var_name='Teknologi', value_name='MW')

            fig_tren = px.line(
                df_melt, x='Tahun', y='MW', color='Teknologi', line_dash='Tipe',
                labels={'Tahun': 'Tahun', 'MW': 'Kapasitas (MW)'},
                template='plotly_dark',
                title='Histori Data (Solid) & Tren Prediksi ML (Putus-Putus)'
            )
            fig_tren.add_vline(x=2025, line_dash='dash', line_color='red', annotation_text='Mulai Prediksi')
            st.plotly_chart(fig_tren, use_container_width=True)

            st.divider()

            # ── Tren Per 5 Tahun + Fluktuasi Bulanan (layout 2 kolom seperti Oji) ──
            col_c1, col_c2 = st.columns(2)

            with col_c1:
                st.write("**Tren Rata-rata Per 5 Tahun (MW):**")
                fig_bar = px.bar(
                    data_5_tahun.reset_index().melt(id_vars="Periode", var_name="Teknologi", value_name="MW"),
                    x="Periode", y="MW", color="Teknologi", barmode="group",
                    labels={"Periode": "Periode"}, template="plotly_dark"
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with col_c2:
                st.write(f"**Fluktuasi Bulanan Tahun {tahun_evaluasi}:**")
                data_tahun_terakhir = data_ml_bulan[data_ml_bulan.index.year == tahun_evaluasi].copy()
                if not data_tahun_terakhir.empty:
                    data_tahun_terakhir.index = data_tahun_terakhir.index.month_name()
                    fig_line = px.line(
                        data_tahun_terakhir.reset_index().melt(
                            id_vars="index", var_name="Teknologi", value_name="MW"
                        ),
                        x="index", y="MW", color="Teknologi",
                        labels={"index": "Bulan"}, template="plotly_dark"
                    )
                    st.plotly_chart(fig_line, use_container_width=True)
                else:
                    st.warning(f"Data bulanan untuk tahun {tahun_evaluasi} tidak tersedia.")

        # ====================================================
        # TAB 2: MCDM & KEBIJAKAN
        # ====================================================
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
                w_tech = st.slider("Potensi Daya (Benefit)",     0.0, 1.0, 0.4)
                w_econ = st.slider("Biaya Investasi (Cost)",      0.0, 1.0, 0.3)
                w_env  = st.slider("Emisi Karbon (Cost)",         0.0, 1.0, 0.2) * pengali_emisi
                w_soc  = st.slider("Penerimaan Sosial (Benefit)", 0.0, 1.0, 0.1)
                total  = w_tech + w_econ + w_env + w_soc
                if total == 0:
                    st.warning("Total bobot nol. Sesuaikan slider.")
                    st.stop()
                w_tech, w_econ, w_env, w_soc = w_tech/total, w_econ/total, w_env/total, w_soc/total

            with col_b2:
                # Ambil prediksi untuk tahun evaluasi (aman dari KeyError)
                skor_daya = get_prediction_for_year(data_ml_tahunan, tahun_evaluasi)

                data_aktual = pd.DataFrame({
                    "Daya Prediksi (MW)":              skor_daya.values,
                    "Investasi Terinflasi (M IDR/MW)": capex_terinflasi,
                    "Emisi (Ton CO2e/GWh)":            emisi_aktual,
                    "Sosial (1-100)":                  sosial_aktual,
                }, index=data_historis.columns)
                st.dataframe(data_aktual.style.format("{:.2f}"))

            # ── Normalisasi & Skor MCDM ──────────────────────────────────────
            norm_df = pd.DataFrame(index=data_aktual.index)
            norm_df["Daya"]     = data_aktual["Daya Prediksi (MW)"] / data_aktual["Daya Prediksi (MW)"].max()
            norm_df["Sosial"]   = data_aktual["Sosial (1-100)"] / data_aktual["Sosial (1-100)"].max()
            norm_df["Investasi"]= data_aktual["Investasi Terinflasi (M IDR/MW)"].min() / data_aktual["Investasi Terinflasi (M IDR/MW)"]
            norm_df["Emisi"]    = data_aktual["Emisi (Ton CO2e/GWh)"].min() / data_aktual["Emisi (Ton CO2e/GWh)"]

            skor_akhir = (
                norm_df["Daya"]     * w_tech +
                norm_df["Investasi"]* w_econ +
                norm_df["Emisi"]    * w_env  +
                norm_df["Sosial"]   * w_soc
            )
            hasil_df = pd.DataFrame(skor_akhir, columns=["Skor Preferensi"]).sort_values(
                "Skor Preferensi", ascending=False
            )

            st.divider()
            st.subheader(f"🏆 Rekomendasi Prioritas Tahun {tahun_evaluasi}")

            col_r1, col_r2 = st.columns(2)
            with col_r1:
                fig_rank = px.bar(
                    hasil_df.reset_index().rename(columns={"index": "Teknologi"}),
                    x="Skor Preferensi", y="Teknologi", orientation="h",
                    color="Skor Preferensi", color_continuous_scale="Teal",
                    template="plotly_dark", labels={"Teknologi": "Teknologi"}
                )
                fig_rank.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig_rank, use_container_width=True)

            with col_r2:
                st.success(
                    f"**Pemenang:** {hasil_df.index[0]} | "
                    f"Skor: {hasil_df.iloc[0]['Skor Preferensi']:.3f}"
                )

                # ── Radar Chart: kontribusi tertimbang per dimensi ──
                bobot_label = {"Daya": w_tech, "Investasi": w_econ, "Emisi": w_env, "Sosial": w_soc}
                categories  = list(bobot_label.keys())
                weighted_df = pd.DataFrame(
                    {dim: norm_df[dim] * bobot_label[dim] for dim in categories},
                    index=norm_df.index
                )

                fig_radar = go.Figure()
                for alt in weighted_df.index:
                    vals = weighted_df.loc[alt, categories].tolist()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=vals + [vals[0]],
                        theta=[f"{d} ({bobot_label[d]*100:.0f}%)" for d in categories] +
                              [f"{categories[0]} ({bobot_label[categories[0]]*100:.0f}%)"],
                        fill='toself', name=alt
                    ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(
                        visible=True,
                        range=[0, max(w_tech, w_econ, w_env, w_soc) * 1.1]
                    )),
                    template="plotly_dark", height=320,
                    margin=dict(l=20, r=20, t=40, b=20),
                    title=dict(text="Kontribusi Bobot per Dimensi", font=dict(size=13))
                )
                st.plotly_chart(fig_radar, use_container_width=True)
                st.caption("💡 Geser slider bobot di kiri — grafik kompas akan ikut berubah.")

            # ── AI Executive Summary ──────────────────────────────────────────
            st.divider()
            st.subheader("🤖 AI Executive Summary")

            col_ai1, col_ai2 = st.columns([1, 3])
            with col_ai1:
                generate_btn = st.button("✨ Generate AI Insight")
            with col_ai2:
                if st.session_state.ai_result_model:
                    st.caption(f"Terakhir dianalisis via: `{st.session_state.ai_result_model}`")

            if generate_btn:
                if not api_key:
                    st.warning("⚠️ Masukkan API Key di sidebar.")
                else:
                    try:
                        client = genai.Client(api_key=api_key)

                        # Hitung skor untuk semua skenario (untuk chart komparasi)
                        skenario_list = [
                            "Business as Usual (BAU)",
                            "Pajak Karbon Tinggi (Pro-Lingkungan)",
                            "Subsidi Masif EBT (Pro-Ekonomi)"
                        ]
                        chart_rows = []
                        for sk in skenario_list:
                            cap_tmp = capex_awal * ((1 + inflasi) ** selisih_tahun)
                            em_mult = 1.5 if sk == "Pajak Karbon Tinggi (Pro-Lingkungan)" else 1.0
                            if sk == "Subsidi Masif EBT (Pro-Ekonomi)":
                                cap_tmp *= 0.7
                            n_inv = cap_tmp.min() / cap_tmp
                            n_em  = emisi_aktual.min() / (emisi_aktual * em_mult)
                            n_d   = skor_daya.values / skor_daya.values.max()
                            n_s   = sosial_aktual / sosial_aktual.max()
                            sc    = n_d * w_tech + n_inv * w_econ + n_em * w_env + n_s * w_soc
                            for alt, s in zip(data_historis.columns, sc):
                                chart_rows.append({"Skenario": sk, "Teknologi": alt, "Skor": round(float(s), 3)})
                        st.session_state.ai_chart_data = pd.DataFrame(chart_rows)

                        prompt_ai = f"""
Anda adalah pakar transisi energi Kabupaten Kulon Progo. Berikan ringkasan SINGKAT (maks 4 kalimat per poin) berformat JSON:
{{
  "mengapa_menang": "...",
  "dampak_ekonomi": "...",
  "rekomendasi": "..."
}}

Data:
- Tahun: {tahun_evaluasi}, Skenario: {kebijakan}, Inflasi: {inflasi*100:.1f}%
- Ranking: {hasil_df.to_string()}
Jawab HANYA dengan JSON valid, tanpa markdown/backtick.
"""
                        with st.spinner("AI sedang menganalisis..."):
                            response = client.models.generate_content(
                                model="gemini-2.5-flash", contents=prompt_ai
                            )
                            raw = response.text.strip()
                            if raw.startswith("```"):
                                raw = raw.split("```")[1]
                                if raw.startswith("json"):
                                    raw = raw[4:]
                            st.session_state.ai_result_text  = json.loads(raw)
                            st.session_state.ai_result_model = "gemini-2.5-flash"

                    except Exception as e:
                        st.error(f"Kesalahan AI: {e}")

            # Tampilkan hasil AI (tetap ada walau pindah tab)
            if st.session_state.ai_result_text:
                res = st.session_state.ai_result_text
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.info(f"🏆 **Mengapa Menang?**\n\n{res.get('mengapa_menang','')}")
                with c2:
                    st.warning(f"💰 **Dampak Ekonomi**\n\n{res.get('dampak_ekonomi','')}")
                with c3:
                    st.success(f"📋 **Rekomendasi**\n\n{res.get('rekomendasi','')}")

            # Chart sensitivitas antar skenario
            if st.session_state.ai_chart_data is not None:
                st.divider()
                st.subheader("📊 Perubahan Skor Antar Skenario Kebijakan")
                fig_sc = px.bar(
                    st.session_state.ai_chart_data,
                    x="Teknologi", y="Skor", color="Skenario",
                    barmode="group", template="plotly_dark",
                    color_discrete_sequence=["#4fc3f7", "#ff8a65", "#81c784"],
                    title="Sensitivitas Skor Preferensi terhadap Perubahan Kebijakan"
                )
                st.plotly_chart(fig_sc, use_container_width=True)

        # ====================================================
        # TAB 3: PETA SPASIAL
        # ====================================================
        with tab3:
            st.header("🗺️ Pemetaan Geospasial Potensi EBT Kulon Progo")

            mode_peta = st.radio(
                "Mode Tampilan Peta:",
                ["🗺️ 2D Interaktif", "🏔️ 3D Globe (Elevasi)"],
                horizontal=True
            )

            if mode_peta == "🗺️ 2D Interaktif":
                m = folium.Map(location=[-7.8288, 110.1587], zoom_start=11, tiles="CartoDB dark_matter")
                for _, row in df_lokasi.iterrows():
                    popup_html = f"""
                    <div style='font-family:sans-serif;min-width:220px'>
                        <h4 style='color:#00bcd4;margin:0'>{row['Teknologi']}</h4>
                        <p style='margin:4px 0'><b>📍</b> {row['Kecamatan']}</p>
                        <hr style='margin:6px 0'>
                        <p style='margin:2px 0'>⚡ Potensi: <b>{row['Potensi_MW']} MW</b></p>
                        <p style='margin:2px 0'>💰 Investasi Awal: <b>Rp {row['Investasi_M_IDR']} M/MW</b></p>
                        <p style='margin:2px 0'>📅 Waktu Optimal: <b>{row['Waktu_Optimal']}</b></p>
                        <p style='margin:2px 0'>🏛️ APBN: <b>{row['APBN_Tahap']}</b></p>
                        <p style='margin:4px 0;font-size:11px;color:#888'>{row['Alasan']}</p>
                    </div>
                    """
                    folium.Marker(
                        location=[row['Lat'], row['Lon']],
                        popup=folium.Popup(popup_html, max_width=260),
                        tooltip=f"{row['Teknologi']} — {row['Potensi_MW']} MW | Klik untuk detail",
                        icon=folium.Icon(color=row['Color'], icon=row['Icon'], prefix='fa')
                    ).add_to(m)
                st_folium(m, width=None, height=480, use_container_width=True)

            else:
                # 3D Globe dengan Plotly Scatter3D
                color_map = {
                    "PLTMH (Air)": "#29b6f6",
                    "PLTS (Surya)": "#ffa726",
                    "PLTB (Angin)": "#66bb6a",
                    "Biomassa": "#ef5350"
                }
                fig_3d = go.Figure()
                for _, row in df_lokasi.iterrows():
                    color = color_map.get(row['Teknologi'], "#ffffff")
                    fig_3d.add_trace(go.Scatter3d(
                        x=[row['Lon']], y=[row['Lat']], z=[row['Elevasi']],
                        mode='markers+text',
                        marker=dict(
                            size=row['Potensi_MW'] * 0.8,
                            color=color, opacity=0.85,
                            symbol='circle',
                            line=dict(color='white', width=1)
                        ),
                        text=[f"{row['Teknologi']}<br>{row['Potensi_MW']} MW"],
                        textposition='top center',
                        hovertemplate=(
                            f"<b>{row['Teknologi']}</b><br>"
                            f"📍 {row['Kecamatan']}<br>"
                            f"⚡ {row['Potensi_MW']} MW<br>"
                            f"💰 Rp {row['Investasi_M_IDR']} M/MW<br>"
                            f"📅 {row['Waktu_Optimal']}<br>"
                            f"🏛️ {row['APBN_Tahap']}<extra></extra>"
                        ),
                        name=row['Teknologi']
                    ))

                fig_3d.update_layout(
                    scene=dict(
                        xaxis_title="Longitude", yaxis_title="Latitude", zaxis_title="Elevasi (m)",
                        bgcolor="rgb(10,15,30)",
                        xaxis=dict(gridcolor="#1e3a5f", backgroundcolor="rgb(10,15,30)"),
                        yaxis=dict(gridcolor="#1e3a5f", backgroundcolor="rgb(10,15,30)"),
                        zaxis=dict(gridcolor="#1e3a5f", backgroundcolor="rgb(10,15,30)"),
                        camera=dict(eye=dict(x=1.5, y=-2.0, z=1.2))
                    ),
                    template="plotly_dark", height=520,
                    title="Visualisasi 3D — Ukuran marker = Potensi MW, Tinggi = Elevasi",
                    legend=dict(x=0, y=1)
                )
                st.plotly_chart(fig_3d, use_container_width=True)
                st.caption("💡 Klik & drag untuk memutar. Scroll untuk zoom. Hover untuk detail.")

            # ── Tabel Detail Investasi & APBN ──────────────────────────────────
            st.divider()
            st.subheader("📋 Detail Potensi, Investasi & Jadwal APBN")

            df_display = df_lokasi[[
                "Teknologi", "Kecamatan", "Potensi_MW",
                "Investasi_M_IDR", "Waktu_Optimal", "APBN_Tahap", "Alasan"
            ]].copy()
            df_display.columns = [
                "Teknologi", "Lokasi", "Potensi (MW)",
                "Investasi Awal (Rp M/MW)", "Waktu Terbaik", "Siklus APBN", "Pertimbangan"
            ]
            st.dataframe(
                df_display.style
                .background_gradient(subset=["Potensi (MW)"], cmap="Blues")
                .background_gradient(subset=["Investasi Awal (Rp M/MW)"], cmap="Oranges"),
                use_container_width=True, height=220
            )

            # ── Timeline APBN Visual ────────────────────────────────────────────
            st.subheader("🗓️ Timeline Optimal Pembangunan vs Siklus APBN")
            timeline_data = []
            for _, r in df_lokasi.iterrows():
                start_y = int(r["Waktu_Optimal"].split("-")[0])
                end_y   = int(r["Waktu_Optimal"].split("-")[1])
                timeline_data.append({
                    "Teknologi": f"{r['Teknologi']} ({r['Kecamatan'].split(' ')[0]})",
                    "Mulai": start_y, "Selesai": end_y,
                    "APBN": r["APBN_Tahap"], "MW": r["Potensi_MW"]
                })
            df_tl = pd.DataFrame(timeline_data)

            fig_tl = px.timeline(
                df_tl.assign(
                    Mulai=pd.to_datetime(df_tl["Mulai"].astype(str) + "-01-01"),
                    Selesai=pd.to_datetime(df_tl["Selesai"].astype(str) + "-12-31")
                ),
                x_start="Mulai", x_end="Selesai", y="Teknologi",
                color="APBN", hover_data=["MW"],
                template="plotly_dark",
                title="Jendela Pembangunan Optimal (sesuai Siklus RPJMN/APBN)",
                color_discrete_sequence=["#4fc3f7", "#ff8a65"]
            )
            fig_tl.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_tl, use_container_width=True)

            st.info(
                "📍 **Catatan:** RPJMN 2025-2029 memprioritaskan proyek EBT skala kecil-menengah "
                "yang shovel-ready. PLTB dan proyek pesisir masuk RPJMN 2030-2034 karena butuh "
                "kajian lingkungan & sosial lebih panjang."
            )

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
        st.exception(e)

else:
    st.info("👈 Upload file Excel, CSV NASA POWER, atau JSON di sidebar untuk memulai.")

    col_ex1, col_ex2 = st.columns(2)

    with col_ex1:
        st.write("**📊 Contoh Format Excel / CSV:**")
        st.dataframe(pd.DataFrame({
            "Tahun":         [2020, 2021, 2022, 2023, 2024],
            "PLTS (Surya)":  [12.5, 13.0, 14.1, 14.8, 15.3],
            "PLTB (Angin)":  [8.2,  8.5,  8.1,  8.7,  9.0],
            "PLTMH (Air)":   [18.0, 17.5, 18.2, 17.9, 18.5],
            "Biomassa":      [10.0, 10.5, 11.0, 11.2, 11.8],
        }))

    with col_ex2:
        st.write("**📋 Contoh Format JSON (dua varian yang didukung):**")
        st.code(
            """// Varian 1 — Array of objects
[
  {"Tahun": 2020, "PLTS (Surya)": 12.5, "PLTB (Angin)": 8.2},
  {"Tahun": 2021, "PLTS (Surya)": 13.0, "PLTB (Angin)": 8.5},
  {"Tahun": 2022, "PLTS (Surya)": 14.1, "PLTB (Angin)": 8.1}
]

// Varian 2 — Object of arrays
{
  "Tahun":        [2020, 2021, 2022],
  "PLTS (Surya)": [12.5, 13.0, 14.1],
  "PLTB (Angin)": [8.2,  8.5,  8.1],
  "PLTMH (Air)":  [18.0, 17.5, 18.2],
  "Biomassa":     [10.0, 10.5, 11.0]
}""",
            language="json"
        )
        st.caption("Kolom **'Tahun'** wajib ada. Kolom lainnya = teknologi EBT (bebas nama).")
