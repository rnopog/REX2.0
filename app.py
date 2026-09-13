
import io
import json
import math
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from groq import Groq

st.set_page_config(
    page_title="REX | Reservoir Engineering eXpert",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEMO_DATA = pd.DataFrame({
    "Date": pd.date_range("2024-01-01", periods=24, freq="MS"),
    "Oil_Rate_bpd": [1000, 980, 965, 950, 930, 910, 890, 865, 845, 820, 800, 775,
                     750, 730, 705, 680, 655, 630, 610, 590, 565, 545, 525, 505],
    "Gas_Rate_mscf_d": [1800, 1810, 1820, 1835, 1850, 1860, 1880, 1900, 1920, 1950, 1980, 2010,
                        2040, 2070, 2100, 2140, 2180, 2210, 2250, 2290, 2330, 2370, 2410, 2450],
    "Water_Rate_bpd": [80, 82, 85, 90, 95, 105, 115, 130, 145, 165, 185, 210,
                       240, 270, 300, 335, 370, 410, 450, 490, 535, 580, 625, 670],
    "Pressure_psia": [3500, 3470, 3440, 3410, 3370, 3330, 3290, 3250, 3200, 3150, 3100, 3040,
                      2980, 2920, 2860, 2790, 2720, 2650, 2580, 2510, 2440, 2370, 2300, 2230],
})

NUMERIC_ALIASES = {
    "oil": ["oil_rate_bpd", "oil_rate", "qo", "q_o", "oil", "oil_production",
            "oil_production_bpd", "oil_rate_stb_day", "oil_bpd", "qoil"],
    "gas": ["gas_rate_mscf_d", "gas_rate", "qg", "q_g", "gas", "gas_production",
            "gas_production_mscf_d", "gas_mscfd", "qgas"],
    "water": ["water_rate_bpd", "water_rate", "qw", "q_w", "water",
              "water_production", "water_production_bpd", "water_bpd", "qwater"],
    "pressure": ["pressure_psia", "pressure", "reservoir_pressure", "p_res",
                 "pres", "reservoir_pressure_psia", "pressure_psi", "pr"],
    "date": ["date", "time", "datetime", "timestamp", "production_date",
             "date_time", "month"],
}

def normalize_name(x):
    return (str(x).strip().lower().replace("-", "_").replace(" ", "_")
            .replace("(", "").replace(")", "").replace("/", "_"))

def find_column(df, aliases):
    normalized = {normalize_name(c): c for c in df.columns}
    for alias in aliases:
        a = normalize_name(alias)
        if a in normalized:
            return normalized[a]
    for c in df.columns:
        nc = normalize_name(c)
        if any(a in nc or nc in a for a in [normalize_name(x) for x in aliases]):
            return c
    return None

def prepare_data(df):
    out = pd.DataFrame(index=df.index)
    mapping = {}
    date_col = find_column(df, NUMERIC_ALIASES["date"])
    if date_col:
        out["Date"] = pd.to_datetime(df[date_col], errors="coerce")
        mapping["Date"] = date_col
    else:
        out["Date"] = np.arange(1, len(df) + 1)

    for key in ["oil", "gas", "water", "pressure"]:
        col = find_column(df, NUMERIC_ALIASES[key])
        if col:
            name = {"oil": "Oil_Rate_bpd", "gas": "Gas_Rate_mscf_d",
                    "water": "Water_Rate_bpd", "pressure": "Pressure_psia"}[key]
            out[name] = pd.to_numeric(df[col], errors="coerce")
            mapping[name] = col

    out = out.dropna(how="all").reset_index(drop=True)
    return out, mapping

def analyze(df):
    a = {}
    if "Oil_Rate_bpd" in df:
        q = df["Oil_Rate_bpd"].dropna()
        if len(q) >= 2:
            a["oil_initial"] = float(q.iloc[0])
            a["oil_current"] = float(q.iloc[-1])
            a["oil_decline_pct"] = float((1 - q.iloc[-1] / q.iloc[0]) * 100) if q.iloc[0] else 0
            positive = q[q > 0]
            if len(positive) >= 3:
                x = np.arange(len(positive), dtype=float)
                slope, intercept = np.polyfit(x, np.log(positive.values), 1)
                a["decline_rate_monthly"] = float(-slope)
                a["decline_rate_annual_pct"] = float((1 - math.exp(12 * slope)) * 100)
                a["forecast_q12"] = float(math.exp(intercept + slope * (len(positive) - 1 + 12)))

    if "Pressure_psia" in df:
        p = df["Pressure_psia"].dropna()
        if len(p) >= 2 and p.iloc[0] != 0:
            a["pressure_initial"] = float(p.iloc[0])
            a["pressure_current"] = float(p.iloc[-1])
            a["pressure_decline_pct"] = float((1 - p.iloc[-1] / p.iloc[0]) * 100)

    if "Water_Rate_bpd" in df:
        w = df["Water_Rate_bpd"]
        a["water_current"] = float(w.dropna().iloc[-1]) if w.notna().any() else 0

    if "Oil_Rate_bpd" in df and "Water_Rate_bpd" in df:
        tmp = df[["Oil_Rate_bpd", "Water_Rate_bpd"]].copy()
        denominator = tmp["Oil_Rate_bpd"] + tmp["Water_Rate_bpd"]
        tmp["WC"] = tmp["Water_Rate_bpd"] / denominator.replace(0, np.nan)
        wc = tmp["WC"].dropna()
        if len(wc):
            a["watercut_initial_pct"] = float(wc.iloc[0] * 100)
            a["watercut_current_pct"] = float(wc.iloc[-1] * 100)

    if "Gas_Rate_mscf_d" in df and "Oil_Rate_bpd" in df:
        tmp = df[["Gas_Rate_mscf_d", "Oil_Rate_bpd"]].dropna()
        if len(tmp):
            oil = tmp.iloc[-1, 1]
            a["gor_current"] = float(tmp.iloc[-1, 0] / oil * 1000) if oil else 0

    return a

def detect_anomalies(df):
    rows = []
    for col, label in [
        ("Oil_Rate_bpd", "Oil rate"),
        ("Pressure_psia", "Pressure"),
        ("Water_Rate_bpd", "Water rate"),
        ("Gas_Rate_mscf_d", "Gas rate"),
    ]:
        if col not in df or df[col].notna().sum() < 6:
            continue
        s = df[col].copy()
        pct = s.pct_change() * 100
        for i in pct.index:
            if pd.notna(pct.loc[i]) and abs(pct.loc[i]) >= 20:
                rows.append({
                    "Index": int(i),
                    "Parameter": label,
                    "Change (%)": round(float(pct.loc[i]), 1),
                    "Severity": "High" if abs(pct.loc[i]) >= 35 else "Medium",
                })

    if rows:
        return pd.DataFrame(rows).drop_duplicates(subset=["Index", "Parameter"])
    return pd.DataFrame(columns=["Index", "Parameter", "Change (%)", "Severity"])

def economics(a, oil_price, opex_monthly, capex, forecast_months):
    qi = a.get("oil_current", 0)
    D = a.get("decline_rate_monthly", 0)
    if qi <= 0:
        return None
    months = np.arange(1, forecast_months + 1)
    q = qi * np.exp(-D * months)
    monthly_bbl = q * 30.4375
    revenue = monthly_bbl * oil_price
    cash = revenue - opex_monthly
    discount = 0.10 / 12
    npv = -capex + np.sum(cash / ((1 + discount) ** months))
    return {
        "forecast_oil_bbl": float(monthly_bbl.sum()),
        "revenue": float(revenue.sum()),
        "opex": float(opex_monthly * forecast_months),
        "npv": float(npv),
    }

def health_score(a, anomalies):
    score = 100
    score -= min(max(a.get("oil_decline_pct", 0), 0) * 0.7, 35)
    score -= min(max(a.get("pressure_decline_pct", 0), 0) * 0.35, 20)
    wc_increase = a.get("watercut_current_pct", 0) - a.get("watercut_initial_pct", 0)
    score -= min(max(wc_increase, 0) * 0.5, 25)
    score -= min(len(anomalies) * 5, 20)
    return max(0, min(100, round(score)))

def build_context(a, anomalies, econ):
    ctx = {
        "metrics": a,
        "anomalies": anomalies.to_dict("records"),
        "economics": econ,
    }
    return json.dumps(ctx, indent=2, default=str)

# -------------------------------------------------
# GROQ / REX
# -------------------------------------------------
def ask_groq(question, context):
    try:
        # Streamlit Cloud Secrets
        api_key = st.secrets["GROQ_API_KEY"]

        # Prevent accidental spaces/newlines in the secret
        api_key = str(api_key).strip()

        if not api_key:
            return "Groq API key is empty. Please check Streamlit Secrets."

        client = Groq(api_key=api_key)

        prompt = f"""
You are REX, a petroleum reservoir-engineering decision-support assistant.
Use ONLY the supplied calculated data. Do not invent measurements.
Clearly distinguish observations from possible interpretations.
Do not claim a definitive reservoir diagnosis from production data alone.
Give concise, professional, engineering-focused answers.

CALCULATED DATA:
{context}

USER QUESTION:
{question}
"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are REX, an AI reservoir engineering assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                },
            ],
            temperature=0.2,
            max_tokens=700,
        )

        return response.choices[0].message.content

    except KeyError:
        return (
            "Groq API key is not configured. "
            "Add GROQ_API_KEY to your Streamlit Secrets."
        )

    except Exception as e:
        return f"Groq request failed: {e}"

# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("🛢️ REX")
st.caption("Reservoir Engineering eXpert • AI-powered reservoir decision support")

with st.sidebar:
    st.header("Data")

    uploaded = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx"],
        key="data_uploader",
        help="Upload your reservoir production data in CSV or XLSX format.",
    )

    use_demo = st.checkbox(
        "Use built-in demo dataset",
        value=False,
        disabled=uploaded is not None,
    )

    st.divider()
    st.header("Economic assumptions")
    oil_price = st.number_input("Oil price ($/bbl)", min_value=0.0, value=70.0, step=5.0)
    opex_monthly = st.number_input("Monthly OPEX ($)", min_value=0.0, value=18000.0, step=1000.0)
    capex = st.number_input("CAPEX ($)", min_value=0.0, value=100000.0, step=10000.0)
    forecast_months = st.slider("Economic forecast (months)", 6, 60, 24)

if uploaded is not None:
    try:
        file_name = uploaded.name.lower()

        if file_name.endswith(".csv"):
            raw = pd.read_csv(uploaded)
        elif file_name.endswith(".xlsx"):
            raw = pd.read_excel(uploaded, engine="openpyxl")
        else:
            st.error("Unsupported file format. Please upload a CSV or XLSX file.")
            st.stop()

        with st.expander("🔍 Uploaded File Information"):
            st.write("**File:**", uploaded.name)
            st.write("**Rows:**", raw.shape[0])
            st.write("**Columns:**", list(raw.columns))
            st.dataframe(raw.head(10), use_container_width=True)

        df, mapping = prepare_data(raw)
        source_name = uploaded.name

        if not mapping:
            st.error(
                "The file was uploaded successfully, but no recognized reservoir "
                "columns were found. Expected columns include Date, Oil Rate, "
                "Gas Rate, Water Rate, and/or Pressure."
            )
            st.info("Check the 'Uploaded File Information' section above to see your column names.")
            st.stop()

    except Exception as e:
        st.error(f"Could not read the uploaded file: {e}")
        st.stop()

elif use_demo:
    raw = DEMO_DATA.copy()
    df, mapping = prepare_data(raw)
    source_name = "Built-in Demo Dataset"

else:
    st.info("Please upload a CSV/XLSX file or select the built-in demo dataset.")
    st.stop()

if len(df) < 2:
    st.error("The dataset needs at least two usable rows.")
    st.stop()

a = analyze(df)
anomalies = detect_anomalies(df)
econ = economics(a, oil_price, opex_monthly, capex, forecast_months)
score = health_score(a, anomalies)

st.success(f"Data loaded: **{source_name}** • {len(df):,} records")

if uploaded is not None:
    with st.expander("🧩 Column Mapping"):
        if mapping:
            mapping_df = pd.DataFrame(
                list(mapping.items()),
                columns=["REX Parameter", "Uploaded Column"],
            )
            st.dataframe(mapping_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No supported columns were detected.")

tabs = st.tabs([
    "🏠 Overview", "📊 Performance", "🚨 Anomalies", "🧠 AI Diagnosis",
    "📉 Forecast", "💰 Economics", "💬 Ask REX"
])

with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reservoir Health", f"{score}/100")
    c2.metric("Current Oil Rate", f"{a.get('oil_current', 0):,.0f} bpd")
    c3.metric("Pressure", f"{a.get('pressure_current', 0):,.0f} psia")
    c4.metric("Water Cut", f"{a.get('watercut_current_pct', 0):.1f}%")

    st.subheader("Key engineering indicators")
    indicators = pd.DataFrame({
        "Indicator": ["Oil decline", "Pressure decline", "Water-cut increase", "Detected anomalies"],
        "Value": [
            f"{a.get('oil_decline_pct', 0):.1f}%",
            f"{a.get('pressure_decline_pct', 0):.1f}%",
            f"{a.get('watercut_current_pct', 0) - a.get('watercut_initial_pct', 0):.1f} percentage points",
            str(len(anomalies)),
        ],
    })
    st.dataframe(indicators, use_container_width=True, hide_index=True)

with tabs[1]:
    st.subheader("Production & pressure performance")

    if "Oil_Rate_bpd" in df:
        st.plotly_chart(
            px.line(df, x="Date", y="Oil_Rate_bpd", markers=True, title="Oil Rate"),
            use_container_width=True,
        )

    if "Pressure_psia" in df:
        st.plotly_chart(
            px.line(df, x="Date", y="Pressure_psia", markers=True, title="Reservoir Pressure"),
            use_container_width=True,
        )

    if "Oil_Rate_bpd" in df and "Water_Rate_bpd" in df:
        chart = df.copy()
        denominator = chart["Oil_Rate_bpd"] + chart["Water_Rate_bpd"]
        chart["Water_Cut_%"] = 100 * chart["Water_Rate_bpd"] / denominator.replace(0, np.nan)
        st.plotly_chart(
            px.line(chart, x="Date", y="Water_Cut_%", markers=True, title="Water Cut"),
            use_container_width=True,
        )

    if "Gas_Rate_mscf_d" in df and "Oil_Rate_bpd" in df:
        chart = df.copy()
        chart["GOR_scf_STB"] = (
            chart["Gas_Rate_mscf_d"] * 1000 /
            chart["Oil_Rate_bpd"].replace(0, np.nan)
        )
        st.plotly_chart(
            px.line(chart, x="Date", y="GOR_scf_STB", markers=True, title="GOR"),
            use_container_width=True,
        )

with tabs[2]:
    st.subheader("Automatic anomaly detection")

    if anomalies.empty:
        st.info("No large period-to-period changes above the current 20% screening threshold were detected.")
    else:
        st.warning(f"{len(anomalies)} potential anomaly/ies detected.")
        st.dataframe(anomalies, use_container_width=True, hide_index=True)
        st.caption(
            "Screening rule: absolute period-to-period change ≥ 20%. "
            "This is a flag for investigation, not proof of a failure mechanism."
        )

with tabs[3]:
    st.subheader("AI Reservoir Diagnosis")
    context = build_context(a, anomalies, econ)

    if st.button("🔎 Generate Engineering Diagnosis", type="primary"):
        with st.spinner("REX is interpreting the calculated indicators..."):
            answer = ask_groq(
                "Provide a concise reservoir performance diagnosis. "
                "Structure it as Observations, Possible Causes, Evidence, "
                "and Recommended Investigation.",
                context,
            )
        st.markdown(answer)
    else:
        st.info("Click the button to generate an AI-assisted interpretation of the calculated results.")

with tabs[4]:
    st.subheader("Production Forecast")

    if "Oil_Rate_bpd" not in df or "oil_current" not in a:
        st.info("Oil-rate data is required for forecasting.")
    else:
        positive = df["Oil_Rate_bpd"].dropna()
        positive = positive[positive > 0]
        x = np.arange(len(positive), dtype=float)

        if len(positive) >= 3:
            slope, intercept = np.polyfit(x, np.log(positive.values), 1)
            horizon = st.slider("Forecast horizon (months)", 6, 60, 24)
            future_x = np.arange(len(positive) + horizon)
            fitted = np.exp(intercept + slope * future_x)

            hist = pd.DataFrame({
                "Period": np.arange(len(positive)),
                "Oil Rate": positive.values,
                "Type": "Historical",
            })
            fc = pd.DataFrame({
                "Period": np.arange(len(positive), len(positive) + horizon),
                "Oil Rate": fitted[len(positive):],
                "Type": "Forecast",
            })

            plot_df = pd.concat([hist, fc], ignore_index=True)

            st.plotly_chart(
                px.line(
                    plot_df,
                    x="Period",
                    y="Oil Rate",
                    color="Type",
                    markers=True,
                    title="Historical + Exponential Decline Forecast",
                ),
                use_container_width=True,
            )

            c1, c2 = st.columns(2)
            c1.metric("Estimated monthly decline", f"{-slope * 100:.2f}%")
            c2.metric(
                "Forecast oil rate at horizon",
                f"{fc['Oil Rate'].iloc[-1]:,.0f} bpd",
            )

with tabs[5]:
    st.subheader("Basic Production Economics")

    if econ:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Forecast Oil", f"{econ['forecast_oil_bbl']:,.0f} bbl")
        c2.metric("Revenue", f"${econ['revenue']:,.0f}")
        c3.metric("OPEX", f"${econ['opex']:,.0f}")
        c4.metric("Simple NPV", f"${econ['npv']:,.0f}")
        st.caption(
            "Illustrative screening economics using the assumptions in the sidebar. "
            "Not a reserves/economic certification."
        )

with tabs[6]:
    st.subheader("💬 Ask REX")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for role, msg in st.session_state.chat:
        with st.chat_message(role):
            st.markdown(msg)

    question = st.chat_input("Ask REX about the reservoir data...")

    if question:
        st.session_state.chat.append(("user", question))

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response = ask_groq(
                    question,
                    build_context(a, anomalies, econ),
                )
            st.markdown(response)
            st.session_state.chat.append(("assistant", response))

with st.expander("📋 Raw data / engineering notes"):
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.download_button(
        "Download processed data",
        df.to_csv(index=False).encode("utf-8"),
        file_name="rex_processed_data.csv",
        mime="text/csv",
    )

    st.markdown("""
**Important:** REX is a prototype decision-support tool. Its anomaly flags, forecasts,
economic calculations, and AI interpretations are screening-level outputs and should
be validated by a qualified engineer using field-specific data and workflows.
""")
