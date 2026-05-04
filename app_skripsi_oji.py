import streamlit as st
import pandas as pd
import numpy as np
import io, json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.express as px

st.set_page_config(page_title="Dashboard EBT Kulon Progo - Oji", layout="wide")
st.title("⚡ Dashboard Analisis Potensi EBT (Hybrid Model)")

# ==========================================
# SIDEBAR
# ==========================================
uploaded_file = st.sidebar.file_uploader("Upload Data", type=["xlsx","csv","json"])

kebijakan = st.sidebar.selectbox(
    "Skenario",
    ["Business as Usual (BAU)",
     "Pajak Karbon Tinggi (Pro-Lingkungan)",
     "Subsidi Masif EBT (Pro-Ekonomi)"]
)

# ==========================================
# HYBRID FUNCTION
# ==========================================
@st.cache_data
def process_data(file_bytes, file_name, kebijakan, tahun_akhir=2060):

    SCENARIO = {
        "Business as Usual (BAU)": (0.03, 2.0),
        "Pajak Karbon Tinggi (Pro-Lingkungan)": (0.06, 3.0),
        "Subsidi Masif EBT (Pro-Ekonomi)": (0.08, 3.5),
    }

    growth_rate, cap_mult = SCENARIO[kebijakan]

    # ================= LOAD DATA =================
    if file_name.endswith(".csv"):
        df = pd.read_csv(io.StringIO(file_bytes.decode("utf-8")))
        df = df.rename(columns={"YEAR":"Tahun"})
        df = df.groupby("Tahun").mean()

    elif file_name.endswith(".json"):
        raw = json.loads(file_bytes.decode("utf-8"))
        df = pd.DataFrame(raw)

        if "Tahun" not in df.columns:
            st.error("JSON harus punya kolom Tahun")
            return None,None,None,None

        df = df.set_index("Tahun")

    else:
        df = pd.read_excel(io.BytesIO(file_bytes)).set_index("Tahun")

    df.index = df.index.astype(int)

    # ================= HYBRID MODEL =================
    tahun_pred = np.arange(2025, tahun_akhir+1)
    X_train = df.index.values.reshape(-1,1)
    X_pred  = tahun_pred.reshape(-1,1)

    df_pred = pd.DataFrame(index=tahun_pred)
    metrics = {}

    for col in df.columns:

        y = df[col].values

        # kalau data terlalu sedikit → skip aman
        if len(y) < 2:
            df_pred[col] = np.repeat(y.mean(), len(tahun_pred))
            metrics[col] = {"R2":0.0,"MAE":0.0}
            continue

        # --- Linear trend
        model = LinearRegression()
        model.fit(X_train, y)
        trend = model.predict(X_pred)

        # --- Growth policy
        growth = np.exp(growth_rate * (tahun_pred - 2025))

        hybrid = trend * growth

        # --- Constraint
        max_cap = np.max(y) * cap_mult
        hybrid = np.clip(hybrid, 0, max_cap)

        # --- Noise kecil
        hybrid += np.random.normal(0, np.std(y)*0.03, len(hybrid))
        hybrid = np.maximum(0, hybrid)

        df_pred[col] = hybrid

        # --- Metrics
        y_pred_train = model.predict(X_train)

        metrics[col] = {
            "R2": round(float(r2_score(y, y_pred_train)),3),
            "MAE": round(float(mean_absolute_error(y, y_pred_train)),3)
        }

    return df, df_pred, metrics


# ==========================================
# MAIN
# ==========================================
if uploaded_file:

    df_hist, df_pred, metrics = process_data(
        uploaded_file.getvalue(),
        uploaded_file.name,
        kebijakan
    )

    if df_hist is None:
        st.stop()

    # ================= METRICS =================
    st.subheader("📊 Evaluasi Model")

    if len(metrics)==0:
        st.warning("Metrics tidak tersedia")
    else:
        metrics_df = pd.DataFrame(metrics).T

        st.dataframe(
            metrics_df.style
            .format("{:.3f}")
            .background_gradient(subset=["R2"], cmap="RdYlGn")
            .background_gradient(subset=["MAE"], cmap="RdYlGn_r"),
            use_container_width=True
        )

    # ================= PLOT =================
    st.subheader("📈 Grafik Prediksi Hybrid")

    df_hist_plot = df_hist.copy().reset_index()
    df_hist_plot["Tipe"]="Historis"

    df_pred_plot = df_pred.copy().reset_index()
    df_pred_plot.rename(columns={"index":"Tahun"}, inplace=True)
    df_pred_plot["Tipe"]="Prediksi"

    df_all = pd.concat([df_hist_plot, df_pred_plot])

    fig = px.line(
        df_all.melt(id_vars=["Tahun","Tipe"]),
        x="Tahun", y="value",
        color="variable",
        line_dash="Tipe"
    )

    fig.add_vline(x=2025, line_dash="dash")

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Upload data untuk mulai")
