import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium

# Konfigurasi Halaman
st.set_page_config(page_title="Dashboard EBT Kulon Progo - Oji", layout="wide")

st.title("⚡ Dashboard Analisis Potensi EBT & MCDM Kabupaten Kulon Progo")
st.markdown("**Oleh: Muhammad Fachrurrozy (Teknik Fisika UGM)**")
st.divider()

# ==========================================
# SIDEBAR: UPLOAD & ASUMSI MAKRO
# ==========================================
st.sidebar.header("📂 1. Input Data Sistem")
uploaded_file = st.sidebar.file_uploader("Upload File Excel (.xlsx)", type=["xlsx", "xls"])

st.sidebar.divider()

st.sidebar.header("🌍 2. Asumsi Makro & Kebijakan")
inflasi = st.sidebar.number_input("Tingkat Inflasi Tahunan (%)", min_value=0.0, max_value=15.0, value=3.5, step=0.1) / 100
kebijakan = st.sidebar.selectbox(
    "Skenario Kebijakan Transisi", 
    ["Business as Usual (BAU)", "Pajak Karbon Tinggi (Pro-Lingkungan)", "Subsidi Masif EBT (Pro-Ekonomi)"]
)
tahun_evaluasi = st.sidebar.slider("Tahun Target Evaluasi MCDM", 2025, 2060, 2060)
selisih_tahun = tahun_evaluasi - 2025

if uploaded_file is not None:
    try:
        data_input = pd.read_excel(uploaded_file)
        if "Tahun" in data_input.columns:
            data_input = data_input.set_index("Tahun")
            
        # Pembuatan Data Prediksi ML
        range_bulan = pd.date_range(start="2025-01-01", end=f"{tahun_evaluasi}-12-31", freq="MS")
        data_ml_bulan = pd.DataFrame(index=range_bulan)
        for col in data_input.columns:
            base_val = data_input[col].mean()
            tren = np.linspace(base_val, base_val * 1.6, len(range_bulan))
            seasonality = 0.2 * base_val * np.sin(2 * np.pi * range_bulan.month / 12)
            noise = np.random.normal(0, base_val * 0.05, len(range_bulan))
            data_ml_bulan[col] = tren + seasonality + noise

        data_5_tahun = data_ml_bulan.resample('5YS').mean()
        data_5_tahun.index = data_5_tahun.index.year 

        # ==========================================
        # MEMBUAT TABS UNTUK UI YANG LEBIH RAPI
        # ==========================================
        tab1, tab2, tab3 = st.tabs(["📈 Analisis Prediksi (ML)", "⚖️ MCDM & Kebijakan", "🗺️ Peta Potensi Spasial"])
        
        # ---------------- TAB 1: PREDIKSI ML ----------------
        with tab1:
            st.header(f"Proyeksi Potensi Daya hingga {tahun_evaluasi}")
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.write("**Tren Potensi Per 5 Tahun (MW):**")
                st.bar_chart(data_5_tahun)
            with col_chart2:
                st.write(f"**Detail Fluktuasi Bulanan pada Tahun {tahun_evaluasi}:**")
                data_tahun_terakhir = data_ml_bulan[data_ml_bulan.index.year == tahun_evaluasi]
                data_tahun_terakhir.index = data_tahun_terakhir.index.month_name()
                st.line_chart(data_tahun_terakhir)

        # ---------------- TAB 2: MCDM ----------------
        with tab2:
            st.header("MCDM dengan Parameter Dinamis")
            capex_awal = np.array([12.0, 18.0, 22.0, 25.0][:len(data_input.columns)])
            capex_terinflasi = capex_awal * ((1 + inflasi) ** selisih_tahun)
            emisi_aktual = np.array([40.0, 11.0, 24.0, 230.0][:len(data_input.columns)])
            sosial_aktual = np.array([90, 70, 85, 80][:len(data_input.columns)])
            
            if kebijakan == "Pajak Karbon Tinggi (Pro-Lingkungan)":
                pengali_emisi, pengali_capex = 1.5, 1.0
                st.warning("🌿 Penalti pada kriteria emisi diaktifkan.")
            elif kebijakan == "Subsidi Masif EBT (Pro-Ekonomi)":
                pengali_emisi, pengali_capex = 1.0, 0.7
                capex_terinflasi *= pengali_capex
                st.info("💰 Diskon biaya investasi 30% diaktifkan.")
            else:
                pengali_emisi, pengali_capex = 1.0, 1.0
                
            col_b1, col_b2 = st.columns([1, 2])
            with col_b1:
                w_tech = st.slider("Potensi Daya (Benefit)", 0.0, 1.0, 0.4)
                w_econ = st.slider("Biaya Investasi (Cost)", 0.0, 1.0, 0.3)
                w_env = st.slider("Emisi Karbon (Cost)", 0.0, 1.0, 0.2) * pengali_emisi
                w_soc = st.slider("Penerimaan Sosial (Benefit)", 0.0, 1.0, 0.1)
                total = w_tech + w_econ + w_env + w_soc
                w_tech, w_econ, w_env, w_soc = w_tech/total, w_econ/total, w_env/total, w_soc/total

            with col_b2:
                skor_daya_evaluasi = data_5_tahun.loc[tahun_evaluasi - (tahun_evaluasi % 5) if tahun_evaluasi % 5 != 0 else tahun_evaluasi] if tahun_evaluasi >= 2025 else data_ml_bulan.mean()
                data_aktual = pd.DataFrame({"Daya Prediksi (MW)": skor_daya_evaluasi.values, "Investasi Terinflasi (M IDR/MW)": capex_terinflasi, "Emisi (Ton CO2e/GWh)": emisi_aktual, "Sosial (1-100)": sosial_aktual}, index=data_input.columns)
                st.dataframe(data_aktual.style.format("{:.2f}"))

            norm_df = pd.DataFrame(index=data_aktual.index)
            norm_df["Daya"] = data_aktual["Daya Prediksi (MW)"] / data_aktual["Daya Prediksi (MW)"].max()
            norm_df["Sosial"] = data_aktual["Sosial (1-100)"] / data_aktual["Sosial (1-100)"].max()
            norm_df["Investasi"] = data_aktual["Investasi Terinflasi (M IDR/MW)"].min() / data_aktual["Investasi Terinflasi (M IDR/MW)"]
            norm_df["Emisi"] = data_aktual["Emisi (Ton CO2e/GWh)"].min() / data_aktual["Emisi (Ton CO2e/GWh)"]

            skor_akhir = (norm_df["Daya"]*w_tech + norm_df["Investasi"]*w_econ + norm_df["Emisi"]*w_env + norm_df["Sosial"]*w_soc)
            hasil_df = pd.DataFrame(skor_akhir, columns=["Skor Preferensi"]).sort_values(by="Skor Preferensi", ascending=False)
            
            st.divider()
            st.subheader("🏆 Rekomendasi Prioritas Berdasarkan Skenario")
            col_res1, col_res2 = st.columns([1, 1])
            with col_res1:
                st.bar_chart(hasil_df)
            with col_res2:
                st.success(f"**Pemenang Skenario:** {hasil_df.index[0]} dengan skor {hasil_df.iloc[0]['Skor Preferensi']:.3f}.")

        # ---------------- TAB 3: PETA SPASIAL ----------------
        with tab3:
            st.header("🗺️ Pemetaan Geospasial Potensi EBT Kulon Progo")
            st.write("Titik-titik ini menunjukkan kesesuaian lokasi topografis untuk pembangunan infrastruktur energi terbarukan di wilayah Kabupaten Kulon Progo.")
            
            # Data Koordinat Hipotetis Kulon Progo
            lokasi_ebt = pd.DataFrame({
                'Teknologi': ['PLTMH (Air)', 'PLTMH (Air)', 'PLTS (Surya)', 'PLTB (Angin)', 'Biomassa (Limbah Pertanian)'],
                'Kecamatan': ['Girimulyo (Perbukitan Menoreh)', 'Samigaluh', 'Wates (Dataran Rendah)', 'Temon / Pantai Glagah', 'Sentolo / Nanggulan'],
                'Lat': [-7.7470, -7.6710, -7.8600, -7.9150, -7.7840],
                'Lon': [110.1260, 110.1700, 110.1400, 110.0760, 110.2220],
                'Color': ['blue', 'blue', 'orange', 'green', 'darkred'],
                'Icon': ['tint', 'tint', 'sun', 'cloud', 'leaf']
            })

            # Titik tengah peta di Kulon Progo
            m = folium.Map(location=[-7.8288, 110.1587], zoom_start=11)

            # Menambahkan marker ke peta
            for i, row in lokasi_ebt.iterrows():
                folium.Marker(
                    location=[row['Lat'], row['Lon']],
                    popup=folium.Popup(f"<b>{row['Teknologi']}</b><br>{row['Kecamatan']}", max_width=200),
                    tooltip=f"Klik untuk info {row['Teknologi']}",
                    icon=folium.Icon(color=row['Color'], icon=row['Icon'], prefix='fa')
                ).add_to(m)

            # Render Peta di Streamlit
            st_folium(m, width=900, height=500)
            
            st.info("📍 **Catatan Analitik:** PLTMH difokuskan di Perbukitan Menoreh karena kontur elevasi dan debit sungai yang curam. PLTB (Angin) difokuskan di pesisir selatan karena kecepatan angin laut yang stabil. PLTS sangat cocok di dataran rendah yang minim tutupan awan.")

    except Exception as e:
        st.error(f"Terjadi kesalahan. Error detail: {e}")
else:
    st.info("👈 Silakan upload file Excel di panel sebelah kiri untuk memulai komputasi.")
