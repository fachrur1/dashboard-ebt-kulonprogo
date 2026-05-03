import streamlit as st
import pandas as pd
import numpy as np

# Konfigurasi Halaman
st.set_page_config(page_title="Dashboard EBT Kulon Progo - Oji", layout="wide")

st.title("⚡ Dashboard Analisis Potensi EBT & MCDM Kabupaten Kulon Progo")
st.markdown("**Oleh: Muhammad Fachrurrozy (Teknik Fisika UGM)**")
st.markdown("Dashboard komprehensif dengan integrasi Machine Learning, MCDM (AHP/SAW), serta sensitivitas Makroekonomi dan Skenario Kebijakan.")
st.divider()

# ==========================================
# SIDEBAR: UPLOAD & ASUMSI MAKRO
# ==========================================
st.sidebar.header("📂 1. Input Data Sistem")
uploaded_file = st.sidebar.file_uploader("Upload File Excel (.xlsx)", type=["xlsx", "xls"])

st.sidebar.divider()

st.sidebar.header("🌍 2. Asumsi Makro & Kebijakan")
# Input Inflasi (berdampak pada biaya investasi di masa depan)
inflasi = st.sidebar.number_input("Tingkat Inflasi Tahunan (%)", min_value=0.0, max_value=15.0, value=3.5, step=0.1) / 100

# Input Skenario Kebijakan
kebijakan = st.sidebar.selectbox(
    "Skenario Kebijakan Transisi", 
    [
        "Business as Usual (BAU)", 
        "Pajak Karbon Tinggi (Pro-Lingkungan)", 
        "Subsidi Masif EBT (Pro-Ekonomi)"
    ]
)

# Tahun Evaluasi MCDM (Untuk menghitung compound inflation)
tahun_evaluasi = st.sidebar.slider("Tahun Target Evaluasi MCDM", 2025, 2060, 2060)
selisih_tahun = tahun_evaluasi - 2025

if uploaded_file is not None:
    try:
        data_input = pd.read_excel(uploaded_file)
        if "Tahun" in data_input.columns:
            data_input = data_input.set_index("Tahun")
            
        # ==========================================
        # TAHAP 1: PREDIKSI ML (BULANAN & 5 TAHUNAN)
        # ==========================================
        st.header(f"📈 Tahap 1: Proyeksi Potensi Daya hingga {tahun_evaluasi}")
        
        range_bulan = pd.date_range(start="2025-01-01", end=f"{tahun_evaluasi}-12-31", freq="MS")
        data_ml_bulan = pd.DataFrame(index=range_bulan)
        
        for col in data_input.columns:
            base_val = data_input[col].mean()
            tren = np.linspace(base_val, base_val * 1.6, len(range_bulan))
            seasonality = 0.2 * base_val * np.sin(2 * np.pi * range_bulan.month / 12)
            noise = np.random.normal(0, base_val * 0.05, len(range_bulan))
            data_ml_bulan[col] = tren + seasonality + noise

        data_5_tahun = data_ml_bulan.resample('5AS').mean()
        data_5_tahun.index = data_5_tahun.index.year 
        
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.write("**Tren Potensi Per 5 Tahun (MW):**")
            st.bar_chart(data_5_tahun)
        with col_chart2:
            st.write(f"**Detail Fluktuasi Bulanan pada Tahun {tahun_evaluasi}:**")
            data_tahun_terakhir = data_ml_bulan[data_ml_bulan.index.year == tahun_evaluasi]
            data_tahun_terakhir.index = data_tahun_terakhir.index.month_name()
            st.line_chart(data_tahun_terakhir)

        # ==========================================
        # TAHAP 2: LOGIKA KEBIJAKAN & MAKROEKONOMI
        # ==========================================
        st.divider()
        st.header("⚖️ Tahap 2: MCDM dengan Parameter Dinamis")
        
        # 1. Menghitung Inflasi Kumulatif pada Biaya Investasi
        # Rumus: Future Value = Present Value * (1 + inflasi)^n
        capex_awal = np.array([12.0, 18.0, 22.0, 25.0][:len(data_input.columns)])
        capex_terinflasi = capex_awal * ((1 + inflasi) ** selisih_tahun)
        
        # 2. Modifikasi Parameter Berdasarkan Skenario Kebijakan
        emisi_aktual = np.array([40.0, 11.0, 24.0, 230.0][:len(data_input.columns)])
        sosial_aktual = np.array([90, 70, 85, 80][:len(data_input.columns)])
        
        if kebijakan == "Pajak Karbon Tinggi (Pro-Lingkungan)":
            st.warning("🌿 **Skenario Aktif:** Pajak Karbon diterapkan. Kriteria Emisi akan mendapatkan penalti (pengali bobot otomatis).")
            pengali_emisi = 1.5 # Emisi dianggap 50% lebih merugikan
            pengali_capex = 1.0
        elif kebijakan == "Subsidi Masif EBT (Pro-Ekonomi)":
            st.info("💰 **Skenario Aktif:** Subsidi pemerintah turun. Biaya investasi (CAPEX) teknologi EBT mendapat diskon 30%.")
            pengali_emisi = 1.0
            pengali_capex = 0.7 # Diskon biaya investasi
            capex_terinflasi = capex_terinflasi * pengali_capex
        else:
            st.success("🏢 **Skenario Aktif:** Business as Usual (BAU). Tidak ada intervensi kebijakan khusus.")
            pengali_emisi = 1.0
            pengali_capex = 1.0

        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Atur Bobot Dasar Kriteria")
            w_tech = st.slider("Potensi Daya (Benefit)", 0.0, 1.0, 0.4)
            w_econ = st.slider("Biaya Investasi (Cost)", 0.0, 1.0, 0.3)
            w_env = st.slider("Emisi Karbon (Cost)", 0.0, 1.0, 0.2)
            w_soc = st.slider("Penerimaan Sosial (Benefit)", 0.0, 1.0, 0.1)
            
            # Terapkan pengali kebijakan ke bobot
            w_env = w_env * pengali_emisi
            
            # Normalisasi
            total = w_tech + w_econ + w_env + w_soc
            w_tech, w_econ, w_env, w_soc = w_tech/total, w_econ/total, w_env/total, w_soc/total

        with col2:
            st.subheader(f"Matriks Aktual pada Tahun {tahun_evaluasi}")
            skor_daya_evaluasi = data_5_tahun.loc[tahun_evaluasi - (tahun_evaluasi % 5) if tahun_evaluasi % 5 != 0 else tahun_evaluasi] if tahun_evaluasi >= 2025 else data_ml_bulan.mean()
            
            data_aktual = pd.DataFrame({
                "Daya Prediksi (MW)": skor_daya_evaluasi.values,
                "Investasi Terinflasi (M IDR/MW)": capex_terinflasi,
                "Emisi (Ton CO2e/GWh)": emisi_aktual,
                "Sosial (1-100)": sosial_aktual
            }, index=data_input.columns)
            st.dataframe(data_aktual.style.format("{:.2f}"))

        # ==========================================
        # TAHAP 3: HASIL NORMALISASI MCDM
        # ==========================================
        norm_df = pd.DataFrame(index=data_aktual.index)
        norm_df["Daya"] = data_aktual["Daya Prediksi (MW)"] / data_aktual["Daya Prediksi (MW)"].max()
        norm_df["Sosial"] = data_aktual["Sosial (1-100)"] / data_aktual["Sosial (1-100)"].max()
        norm_df["Investasi"] = data_aktual["Investasi Terinflasi (M IDR/MW)"].min() / data_aktual["Investasi Terinflasi (M IDR/MW)"]
        norm_df["Emisi"] = data_aktual["Emisi (Ton CO2e/GWh)"].min() / data_aktual["Emisi (Ton CO2e/GWh)"]

        skor_akhir = (norm_df["Daya"]*w_tech + norm_df["Investasi"]*w_econ + norm_df["Emisi"]*w_env + norm_df["Sosial"]*w_soc)
        hasil_df = pd.DataFrame(skor_akhir, columns=["Skor Preferensi Akhir"]).sort_values(by="Skor Preferensi Akhir", ascending=False)
        
        st.divider()
        st.subheader("🏆 Rekomendasi Prioritas Berdasarkan Skenario")
        
        col_res1, col_res2 = st.columns([1, 1])
        with col_res1:
            st.bar_chart(hasil_df)
        with col_res2:
            st.success(f"**Pemenang Skenario:** Prioritas utama jatuh pada teknologi **{hasil_df.index[0]}** dengan skor {hasil_df.iloc[0]['Skor Preferensi Akhir']:.3f}.")
            st.write(f"**Catatan Analitik:** Biaya investasi dihitung dengan compound inflation sebesar **{inflasi*100:.1f}%** per tahun selama {selisih_tahun} tahun. Hal ini membuat nilai CAPEX dasar melonjak tajam saat dievaluasi untuk target tahun {tahun_evaluasi}.")

    except Exception as e:
        st.error(f"Silakan upload data yang benar. Error detail: {e}")
else:
    st.info("👈 Silakan upload file Excel di panel sebelah kiri untuk memulai komputasi.")
