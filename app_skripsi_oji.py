import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from google import genai
import plotly.express as px
import plotly.graph_objects as go
import json

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Dashboard Rencana Skenario EBT Kulon Progo", layout="wide")

st.title("⚡ Dashboard Analisis Potensi EBT & MCDM Kabupaten Kulon Progo")
st.markdown("**Oleh: Muhammad Fachrurrozy (Teknik Fisika UGM)**")
st.divider()

# ==========================================
# SESSION STATE — FIX #1: AI summary tidak hilang saat pindah tab
# ==========================================
if "ai_result_text" not in st.session_state:
    st.session_state.ai_result_text = None
if "ai_result_model" not in st.session_state:
    st.session_state.ai_result_model = None
if "ai_chart_data" not in st.session_state:
    st.session_state.ai_chart_data = None

# ==========================================
# SIDEBAR
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
st.sidebar.divider()

st.sidebar.header("🧠 3. Integrasi AI")
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    st.sidebar.success("✅ AI terhubung via Server Key.")
except Exception:
    api_key = st.sidebar.text_input("Gemini API Key", type="password", placeholder="Masukkan API Key...", help="Dapatkan di aistudio.google.com")

# ==========================================
# DATA LOKASI EBT (dipakai di Tab 3)
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
# MAIN CONTENT
# ==========================================
if uploaded_file is not None:
    try:
        data_input = pd.read_excel(uploaded_file)
        if "Tahun" in data_input.columns:
            data_input = data_input.set_index("Tahun")

        rng = np.random.default_rng(seed=42)
        range_bulan = pd.date_range(start="2025-01-01", end=f"{tahun_evaluasi}-12-31", freq="MS")
        data_ml_bulan = pd.DataFrame(index=range_bulan)

        for col in data_input.columns:
            base_val = data_input[col].mean()
            tren = np.linspace(base_val, base_val * 1.6, len(range_bulan))
            seasonality = 0.2 * base_val * np.sin(2 * np.pi * range_bulan.month / 12)
            noise = rng.normal(0, base_val * 0.05, len(range_bulan))
            data_ml_bulan[col] = tren + seasonality + noise

        data_5_tahun = data_ml_bulan.resample('5YS').mean()
        data_5_tahun.index = data_5_tahun.index.year

        # TABS
        tab1, tab2, tab3 = st.tabs(["📈 Analisis Prediksi (ML)", "⚖️ MCDM & Kebijakan", "🗺️ Peta Potensi Spasial"])

        # ====================================================
        # TAB 1: PREDIKSI ML
        # ====================================================
        with tab1:
            st.header(f"Proyeksi Potensi Daya hingga {tahun_evaluasi}")
            col_c1, col_c2 = st.columns(2)

            with col_c1:
                st.write("**Tren Potensi Per 5 Tahun (MW):**")
                fig_bar = px.bar(
                    data_5_tahun.reset_index().melt(id_vars="index", var_name="Teknologi", value_name="MW"),
                    x="index", y="MW", color="Teknologi", barmode="group",
                    labels={"index": "Tahun"}, template="plotly_dark"
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with col_c2:
                st.write(f"**Fluktuasi Bulanan Tahun {tahun_evaluasi}:**")
                data_tahun_terakhir = data_ml_bulan[data_ml_bulan.index.year == tahun_evaluasi].copy()
                if not data_tahun_terakhir.empty:
                    data_tahun_terakhir.index = data_tahun_terakhir.index.month_name()
                    fig_line = px.line(
                        data_tahun_terakhir.reset_index().melt(id_vars="index", var_name="Teknologi", value_name="MW"),
                        x="index", y="MW", color="Teknologi",
                        labels={"index": "Bulan"}, template="plotly_dark"
                    )
                    st.plotly_chart(fig_line, use_container_width=True)
                else:
                    st.warning(f"Data untuk tahun {tahun_evaluasi} tidak tersedia.")

        # ====================================================
        # TAB 2: MCDM
        # ====================================================
        with tab2:
            st.header("MCDM dengan Parameter Dinamis")

            n_alt = len(data_input.columns)
            capex_awal    = np.array([12.0, 18.0, 22.0, 25.0][:n_alt])
            emisi_aktual  = np.array([40.0, 11.0, 24.0, 230.0][:n_alt])
            sosial_aktual = np.array([90, 70, 85, 80][:n_alt])
            capex_terinflasi = capex_awal * ((1 + inflasi) ** selisih_tahun)

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
                tahun_tersedia = data_5_tahun.index[data_5_tahun.index <= tahun_evaluasi]
                tahun_lookup   = tahun_tersedia[-1] if len(tahun_tersedia) > 0 else data_5_tahun.index[0]
                skor_daya      = data_5_tahun.loc[tahun_lookup]

                data_aktual = pd.DataFrame({
                    "Daya Prediksi (MW)":              skor_daya.values,
                    "Investasi Terinflasi (M IDR/MW)": capex_terinflasi,
                    "Emisi (Ton CO2e/GWh)":            emisi_aktual,
                    "Sosial (1-100)":                  sosial_aktual,
                }, index=data_input.columns)
                st.dataframe(data_aktual.style.format("{:.2f}"))

            norm_df = pd.DataFrame(index=data_aktual.index)
            norm_df["Daya"]     = data_aktual["Daya Prediksi (MW)"] / data_aktual["Daya Prediksi (MW)"].max()
            norm_df["Sosial"]   = data_aktual["Sosial (1-100)"] / data_aktual["Sosial (1-100)"].max()
            norm_df["Investasi"] = data_aktual["Investasi Terinflasi (M IDR/MW)"].min() / data_aktual["Investasi Terinflasi (M IDR/MW)"]
            norm_df["Emisi"]    = data_aktual["Emisi (Ton CO2e/GWh)"].min() / data_aktual["Emisi (Ton CO2e/GWh)"]

            skor_akhir = (
                norm_df["Daya"] * w_tech + norm_df["Investasi"] * w_econ +
                norm_df["Emisi"] * w_env + norm_df["Sosial"] * w_soc
            )
            hasil_df = pd.DataFrame(skor_akhir, columns=["Skor Preferensi"]).sort_values("Skor Preferensi", ascending=False)

            st.divider()
            st.subheader("🏆 Rekomendasi Prioritas")
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                fig_rank = px.bar(
                    hasil_df.reset_index(), x="Skor Preferensi", y="index",
                    orientation="h", color="Skor Preferensi",
                    color_continuous_scale="Teal", template="plotly_dark",
                    labels={"index": "Teknologi"}
                )
                fig_rank.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig_rank, use_container_width=True)
            with col_r2:
                st.success(f"**Pemenang:** {hasil_df.index[0]} | Skor: {hasil_df.iloc[0]['Skor Preferensi']:.3f}")

                # Radar chart perbandingan dimensi
                categories = ["Daya", "Investasi", "Emisi", "Sosial"]
                fig_radar = go.Figure()
                for alt in norm_df.index:
                    vals = norm_df.loc[alt, categories].tolist()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=vals + [vals[0]], theta=categories + [categories[0]],
                        fill='toself', name=alt
                    ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    template="plotly_dark", height=300,
                    margin=dict(l=20, r=20, t=30, b=20)
                )
                st.plotly_chart(fig_radar, use_container_width=True)

            # ==========================================
            # FIX #2: AI INSIGHT — simpan ke session_state agar tidak hilang
            # ==========================================
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

                        # Siapkan data untuk chart perubahan skor antar skenario
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
                            for alt, s in zip(data_input.columns, sc):
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
                            # bersihkan jika ada markdown fence
                            if raw.startswith("```"):
                                raw = raw.split("```")[1]
                                if raw.startswith("json"):
                                    raw = raw[4:]
                            st.session_state.ai_result_text  = json.loads(raw)
                            st.session_state.ai_result_model = "gemini-2.5-flash"

                    except Exception as e:
                        st.error(f"Kesalahan AI: {e}")

            # Tampilkan hasil dari session_state (tetap ada walau pindah tab)
            if st.session_state.ai_result_text:
                res = st.session_state.ai_result_text
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.info(f"🏆 **Mengapa Menang?**\n\n{res.get('mengapa_menang','')}")
                with c2:
                    st.warning(f"💰 **Dampak Ekonomi**\n\n{res.get('dampak_ekonomi','')}")
                with c3:
                    st.success(f"📋 **Rekomendasi**\n\n{res.get('rekomendasi','')}")

            # FIX #2b: Chart perubahan skor antar skenario
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
        # TAB 3: PETA SPASIAL — FIX #3: 3D + Detail Investasi & APBN
        # ====================================================
        with tab3:
            st.header("🏔️ Digital Twin: Terrain Mode Kulon Progo")
            import pydeck as pdk

            # Penjelasan Interaktif
            st.write("Visualisasi ini menggunakan data topografi nyata. Tinggi batang (Column) melambangkan **Potensi Energi (MW)**.")

            # Menyiapkan data untuk Pydeck (Warna dalam format RGB)
            df_pdk = df_lokasi.copy()
            color_pdk = {
                "blue": [41, 182, 246, 200],    # PLTMH
                "orange": [255, 167, 38, 200],  # PLTS
                "green": [102, 187, 106, 200],  # PLTB
                "darkred": [239, 83, 80, 200]   # Biomassa
            }
            df_pdk["color_rgb"] = df_pdk["Color"].map(color_pdk)

            # Konfigurasi View (Sudut Pandang Kamera 3D)
            view_state = pdk.ViewState(
                latitude=-7.8288,
                longitude=110.1587,
                zoom=10.5,
                pitch=50,  # Kemiringan kamera untuk efek 3D
                bearing=-10
            )

            # LAYER 1: Column Layer (Batang Energi)
            # Tinggi batang = Potensi_MW * 200 (agar terlihat proporsional di peta)
            column_layer = pdk.Layer(
                "ColumnLayer",
                data=df_pdk,
                get_position=["Lon", "Lat"],
                get_elevation="Potensi_MW * 200",
                elevation_scale=1,
                radius=400,
                get_fill_color="color_rgb",
                pickable=True,
                auto_highlight=True,
            )

            # LAYER 2: Terrain Layer (Kontur Tanah Nyata)
            # Catatan: Terrain layer terbaik membutuhkan Mapbox Token, 
            # tapi kita bisa menggunakan Terrain RGB open-source
            terrain_layer = pdk.Layer(
                "TerrainLayer",
                elevation_decoder={
                    "rScaler": 1, "gScaler": 0, "bScaler": 0, "offset": 0
                },
                elevation_data="https://assets.mapbox.com/raster-tiles/mapbox.terrain-rgb/{z}/{x}/{y}.pngraw",
                texture="https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
            )

            # Render Peta Pydeck
            r = pdk.Deck(
                layers=[terrain_layer, column_layer],
                initial_view_state=view_state,
                tooltip={
                    "html": """
                        <b>Teknologi:</b> {Teknologi}<br/>
                        <b>Lokasi:</b> {Kecamatan}<br/>
                        <b>Potensi:</b> {Potensi_MW} MW<br/>
                        <b>Investasi:</b> Rp {Investasi_M_IDR} M/MW<br/>
                        <b>Siklus APBN:</b> {APBN_Tahap}
                    """,
                    "style": {"backgroundColor": "steelblue", "color": "white"}
                }
            )

            st.pydeck_chart(r)
            
            st.success("💡 **Analisis Terrain:** Perhatikan bahwa lokasi **PLTMH** berada pada area dengan elevasi tinggi (tekstur pegunungan), yang mengonfirmasi validitas teknis pemilihan lokasi berdasarkan *head* air.")
        

            # ---- Tabel Detail Investasi & APBN ----
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

            # ---- Timeline APBN Visual ----
            st.subheader("🗓️ Timeline Optimal Pembangunan vs Siklus APBN")
            timeline_data = []
            for _, r in df_lokasi.iterrows():
                start_y = int(r["Waktu_Optimal"].split("-")[0])
                end_y   = int(r["Waktu_Optimal"].split("-")[1])
                timeline_data.append({
                    "Teknologi": f"{r['Teknologi']} ({r['Kecamatan'].split(' ')[0]})",
                    "Mulai": start_y, "Selesai": end_y,
                    "APBN": r["APBN_Tahap"],
                    "MW": r["Potensi_MW"]
                })
            df_tl = pd.DataFrame(timeline_data)

            fig_tl = px.timeline(
                df_tl.assign(
                    Mulai=pd.to_datetime(df_tl["Mulai"].astype(str) + "-01-01"),
                    Selesai=pd.to_datetime(df_tl["Selesai"].astype(str) + "-12-31")
                ),
                x_start="Mulai", x_end="Selesai",
                y="Teknologi", color="APBN",
                hover_data=["MW"],
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
    st.info("👈 Upload file Excel di sidebar untuk memulai.")
    st.write("**Contoh Format Excel:**")
    st.dataframe(pd.DataFrame({
        "Tahun": [2020, 2021, 2022],
        "PLTS (Surya)": [12.5, 13.0, 14.1],
        "PLTB (Angin)": [8.2, 8.5, 8.1],
        "PLTMH (Air)":  [18.0, 17.5, 18.2],
        "Biomassa":     [10.0, 10.5, 11.0],
    }))
