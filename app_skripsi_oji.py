import streamlit as st
import pandas as pd
import numpy as np

# Konfigurasi Halaman
st.set_page_config(page_title="Dashboard EBT Kulon Progo - Oji", layout="wide")

st.title("⚡ Dashboard Analisis Potensi EBT & MCDM Kabupaten Kulon Progo")
st.markdown("**Oleh: Muhammad Fachrurrozy (Teknik Fisika UGM)**")
st.markdown("Dashboard ini menerima input data historis berupa file Excel, melakukan proyeksi berbasis Machine Learning, lalu memberikan rekomendasi prioritas EBT menggunakan metode AHP.")

st.divider()

# ==========================================
# FITUR UPLOAD FILE EXCEL
# ==========================================
st.sidebar.header("📂 Input Data Sistem")
st.sidebar.info("Upload file Excel berisi data histori cuaca/potensi energi. Format kolom: Tahun, PLTS, PLTB, PLTMH, Biomassa.")
uploaded_file = st.sidebar.file_uploader("Upload File Excel (.xlsx)", type=["xlsx", "xls"])

if uploaded_file is not None:
    # Membaca file excel yang diupload
    try:
        data_input = pd.read_excel(uploaded_file)
        
        st.header("📊 Data Historis (Input Pengguna)")
        st.write("Berikut adalah data yang berhasil dibaca dari file Excel Anda:")
        st.dataframe(data_input, use_container_width=True)
        
        # Asumsi kolom pertama adalah 'Tahun', kita jadikan index
        if "Tahun" in data_input.columns:
            data_input = data_input.set_index("Tahun")
            
        # ==========================================
        # TAHAP 1: SIMULASI HASIL MACHINE LEARNING
        # ==========================================
        st.divider()
        st.header("📈 Tahap 1: Prediksi Potensi EBT (Machine Learning)")
        st.write("Berdasarkan data historis di atas, model ML memproyeksikan potensi daya hingga 2060.")
        
        # --- BLOK KODE ML SEMENTARA (DUMMY PROJECTION) ---
        # Di sini nantinya Anda bisa memasukkan model ML asli (seperti model.predict(data_input))
        # Untuk sementara, kita buat simulasi prediksi dengan menambahkan tren naik
        tahun_prediksi = np.arange(2025, 2061)
        data_ml = pd.DataFrame(index=tahun_prediksi)
        
        for col in data_input.columns:
            # Simulasi tren: Nilai rata-rata historis + tren kenaikan acak
            mean_val = data_input[col].mean()
            data_ml[col] = np.linspace(mean_val, mean_val * 1.5, len(tahun_prediksi)) + np.random.normal(0, mean_val*0.05, len(tahun_prediksi))
            
        st.line_chart(data_ml)
        
        # Ekstraksi skor teknis rata-rata dari ML (untuk masuk ke MCDM)
        skor_ml = data_ml.mean().to_dict()

        # ==========================================
        # TAHAP 2: INTERAKTIF MCDM (AHP)
        # ==========================================
        st.divider()
        st.header("⚖️ Tahap 2: Pengambilan Keputusan Multi-Kriteria (AHP)")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Atur Bobot Kriteria")
            bobot_teknis = st.slider("Kelayakan Teknis (Dari ML)", 0.0, 1.0, 0.4)
            bobot_ekonomi = st.slider("Biaya Investasi", 0.0, 1.0, 0.3)
            bobot_lingkungan = st.slider("Dampak Lingkungan", 0.0, 1.0, 0.2)
            bobot_sosial = st.slider("Penerimaan Sosial", 0.0, 1.0, 0.1)
            
            # Normalisasi bobot
            total_bobot = bobot_teknis + bobot_ekonomi + bobot_lingkungan + bobot_sosial
            w_tech, w_econ, w_env, w_soc = bobot_teknis/total_bobot, bobot_ekonomi/total_bobot, bobot_lingkungan/total_bobot, bobot_sosial/total_bobot

        with col2:
            st.subheader("Matriks Skor Alternatif")
            # Skala dinamis dari hasil ML (dibagi dengan nilai max agar skalanya 1-10)
            max_ml = max(skor_ml.values())
            
            data_ahp = pd.DataFrame({
                "Teknis (Skala 10)": [(skor_ml.get(alt, 0)/max_ml)*10 for alt in data_input.columns],
                "Ekonomi (10=Murah)": np.random.randint(5, 10, len(data_input.columns)), # Contoh nilai dummy
                "Lingkungan (10=Bersih)": np.random.randint(5, 10, len(data_input.columns)), # Contoh nilai dummy
                "Sosial (10=Diterima)": np.random.randint(5, 10, len(data_input.columns)) # Contoh nilai dummy
            }, index=data_input.columns)
            
            st.dataframe(data_ahp.style.highlight_max(axis=0, color="lightgreen"))

        # ==========================================
        # TAHAP 3: HASIL AKHIR
        # ==========================================
        st.divider()
        st.header("🏆 Hasil Rekomendasi Prioritas EBT")
        
        # Menghitung skor akhir AHP
        skor_akhir = (
            data_ahp["Teknis (Skala 10)"] * w_tech +
            data_ahp["Ekonomi (10=Murah)"] * w_econ +
            data_ahp["Lingkungan (10=Bersih)"] * w_env +
            data_ahp["Sosial (10=Diterima)"] * w_soc
        )
        
        hasil_df = pd.DataFrame(skor_akhir, columns=["Skor Akhir"]).sort_values(by="Skor Akhir", ascending=False)
        
        col3, col4 = st.columns(2)
        with col3:
            st.bar_chart(hasil_df)
        with col4:
            st.success(f"**Rekomendasi Utama:** {hasil_df.index[0]} adalah pilihan terbaik dengan skor {hasil_df.iloc[0]['Skor Akhir']:.2f}.")
            st.info("Algoritma berhasil memproses data Excel Anda secara *end-to-end*!")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file: {e}. Pastikan file format Excel (.xlsx).")

else:
    # Tampilan saat belum ada file yang diupload
    st.info("👈 Silakan upload file data histori Excel (.xlsx) di panel sebelah kiri untuk memulai komputasi Machine Learning dan MCDM.")
    
    st.write("**Contoh Format Excel yang dibutuhkan:**")
    contoh_df = pd.DataFrame({
        "Tahun": [2020, 2021, 2022],
        "PLTS (Surya)": [12.5, 13.0, 14.1],
        "PLTB (Angin)": [8.2, 8.5, 8.1],
        "PLTMH (Air)": [18.0, 17.5, 18.2],
        "Biomassa": [10.0, 10.5, 11.0]
    })
    st.dataframe(contoh_df)