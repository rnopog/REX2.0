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

# ============================================================
# CUSTOM PROFESSIONAL UI / CSS STYLING
# ============================================================
st.markdown("""
<style>
    /* Main Theme & Background */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Headers & Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* Metric Cards Styling */
    div[data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 15px 20px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        border-color: #58a6ff;
        transform: translateY(-2px);
    }
    div[data-testid="stMetric"] label {
        color: #8b949e !important;
        font-weight: 500 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #58a6ff !important;
        font-weight: 700 !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #161b22;
        padding: 8px;
        border-radius: 8px;
        border: 1px solid #30363d;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        background-color: transparent;
        border-radius: 6px;
        color: #8b949e;
        font-weight: 500;
        border: none;
        padding: 0 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #238636 !important;
        color: #ffffff !important;
    }

    /* Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #30363d;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #238636;
        color: white;
        border-radius: 6px;
        font-weight: 600;
        border: none;
        padding: 0.5rem 1rem;
        transition: background-color 0.2s;
    }
    .stButton button:hover {
        background-color: #2ea043;
    }
    
    /* Dataframes & Tables */
    div[data-testid="stDataFrame"] {
        border: 1px solid #30363d;
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

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
    "Condensate_Rate_bpd": [45, 44, 44, 43, 42, 42, 41, 40, 40, 39, 38, 38,
                            37, 36, 35, 35, 34, 33, 32, 32, 31, 30, 29, 29],
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
    "cum_gas": ["cum_gas_mmscf", "cumulative_gas_mmscf", "gp", "cum_gas",
                "cumulative_gas_production", "gas_cum", "cum_production_gas",
                "cumulative_gas"],
    "condensate": ["condensate_rate_bpd", "condensate_rate", "condensate",
                   "cond_rate", "ngl_rate", "condensate_bpd", "cgr_bpd",
                   "condensate_production"],
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

    key_to_name = {
        "oil": "Oil_Rate_bpd",
        "gas": "Gas_Rate_mscf_d",
        "water": "Water_Rate_bpd",
        "pressure": "Pressure_psia",
        "cum_gas": "Cum_Gas_MMscf",
        "condensate": "Condensate_Rate_bpd",
    }

    for key in ["oil", "gas", "water", "pressure", "cum_gas", "condensate"]:
        col = find_column(df, NUMERIC_ALIASES[key])
        if col:
            name = key_to_name[key]
            out[name] = pd.to_numeric(df[col], errors="coerce")
            mapping[name] = col

    out = out.dropna(how="all").reset_index(drop=True)
    return out, mapping

# ============================================================
# GAS PVT CORRELATIONS
# ============================================================
R_CONST = 10.732   # psia*ft3 / (lb-mol*R)
PSC = 14.696       # psia
TSC = 520.0        # deg R (60 F)

def pseudocritical_properties(sg_gas, is_condensate=False):
    if is_condensate:
        Tpc = 187 + 330 * sg_gas - 71.5 * sg_gas ** 2
        Ppc = 706 - 51.7 * sg_gas - 11.1 * sg_gas ** 2
    else:
        Tpc = 168 + 325 * sg_gas - 12.5 * sg_gas ** 2
        Ppc = 677 + 15.0 * sg_gas - 37.5 * sg_gas ** 2
    return Tpc, Ppc

def z_factor_dak(Tpr, Ppr, tol=1e-8, max_iter=100):
    A1, A2, A3, A4, A5 = 0.3265, -1.0700, -0.5339, 0.01569, -0.05165
    A6, A7, A8, A9, A10, A11 = 0.5475, -0.7361, 0.1844, 0.1056, 0.6134, 0.7210

    def f(z):
        rho_r = 0.27 * Ppr / (z * Tpr)
        return (1
                + (A1 + A2 / Tpr + A3 / Tpr ** 3 + A4 / Tpr ** 4 + A5 / Tpr ** 5) * rho_r
                + (A6 + A7 / Tpr + A8 / Tpr ** 2) * rho_r ** 2
                - A9 * (A7 / Tpr + A8 / Tpr ** 2) * rho_r ** 5
                + A10 * (1 + A11 * rho_r ** 2) * (rho_r ** 2 / Tpr ** 3) * math.exp(-A11 * rho_r ** 2)
                - z)

    lo, hi = 0.2, 3.0
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        return 0.90
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        fm = f(mid)
        if abs(fm) < tol:
            return mid
        if flo * fm < 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return (lo + hi) / 2

def gas_fvf_rcf_scf(T_R, P_psia, Z):
    if P_psia <= 0:
        return np.nan
    return 0.02827 * Z * T_R / P_psia

def gas_viscosity_cp(T_R, P_psia, Z, sg_gas):
    if Z <= 0 or T_R <= 0 or P_psia <= 0:
        return np.nan
    M = 28.97 * sg_gas
    rho_g = (P_psia * M) / (Z * R_CONST * T_R)
    K = ((9.4 + 0.02 * M) * T_R ** 1.5) / (209 + 19 * M + T_R)
    X = 3.5 + 986.0 / T_R + 0.01 * M
    Y = 2.4 - 0.2 * X
    return K * math.exp(X * (rho_g / 62.4) ** Y) * 1e-4

def compute_gas_pvt(df, sg_gas, res_temp_F):
    if "Pressure_psia" not in df:
        return None
    T_R = res_temp_F + 459.67
    Tpc, Ppc = pseudocritical_properties(sg_gas)
    Tpr = T_R / Tpc
    rows = []
    for _, p in df["Pressure_psia"].items():
        if pd.isna(p) or p <= 0:
            rows.append({"Z": np.nan, "Bg_rcf_scf": np.nan, "Gas_Viscosity_cp": np.nan, "Ppr": np.nan})
            continue
        Ppr = p / Ppc
        z = z_factor_dak(Tpr, Ppr)
        bg = gas_fvf_rcf_scf(T_R, p, z)
        mu = gas_viscosity_cp(T_R, p, z, sg_gas)
        rows.append({"Z": z, "Bg_rcf_scf": bg, "Gas_Viscosity_cp": mu, "Ppr": Ppr})
    pvt = pd.DataFrame(rows, index=df.index)
    pvt["Tpc_R"], pvt["Ppc_psia"], pvt["Tpr"] = Tpc, Ppc, Tpr
    return pvt

def estimate_cum_gas(df):
    if "Cum_Gas_MMscf" in df and df["Cum_Gas_MMscf"].notna().sum() >= 2:
        return df["Cum_Gas_MMscf"]
    if "Gas_Rate_mscf_d" not in df:
        return None
    rate = df["Gas_Rate_mscf_d"].fillna(0)
    days_per_period = 30.4375
    incr_mmscf = rate * days_per_period / 1000.0
    return incr_mmscf.cumsum()

def pz_material_balance(df, sg_gas, res_temp_F):
    pvt = compute_gas_pvt(df, sg_gas, res_temp_F)
    if pvt is None:
        return None, None
    cum_gas = estimate_cum_gas(df)
    if cum_gas is None:
        return None, None

    pz = pd.DataFrame({
        "Date": df["Date"] if "Date" in df else np.arange(len(df)),
        "Pressure_psia": df["Pressure_psia"],
        "Z": pvt["Z"],
        "Cum_Gas_MMscf": cum_gas,
    })
    pz["P_over_Z"] = pz["Pressure_psia"] / pz["Z"]
    pz = pz.dropna(subset=["Pressure_psia", "Z", "Cum_Gas_MMscf", "P_over_Z"])

    if len(pz) < 3:
        return pz, None

    x = pz["Cum_Gas_MMscf"].values
    y = pz["P_over_Z"].values
    slope, intercept = np.polyfit(x, y, 1)
    ogip = -intercept / slope if slope != 0 else np.nan
    fitted_line = slope * x + intercept
    ss_res = float(np.sum((y - fitted_line) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0

    result = {
        "slope": float(slope),
        "intercept": float(intercept),
        "OGIP_MMscf": float(ogip) if ogip == ogip and ogip > 0 else None,
        "OGIP_Bscf": float(ogip / 1000) if ogip == ogip and ogip > 0 else None,
        "r_squared": float(r2),
        "cum_gas_produced_MMscf": float(x[-1]),
        "recovery_factor_pct": None,
    }
    if result["OGIP_MMscf"]:
        result["recovery_factor_pct"] = float(x[-1] / result["OGIP_MMscf"] * 100)
    return pz, result

def rawlins_schellhardt(rates_mscfd, pwf_psia, pr_psia):
    rates = np.asarray(rates_mscfd, dtype=float)
    pwf = np.asarray(pwf_psia, dtype=float)
    delta_p2 = pr_psia ** 2 - pwf ** 2
    mask = (delta_p2 > 0) & (rates > 0) & np.isfinite(delta_p2) & np.isfinite(rates)
    if mask.sum() < 2:
        return None
    x = np.log10(delta_p2[mask])
    y = np.log10(rates[mask])
    n, logC = np.polyfit(x, y, 1)
    n = float(np.clip(n, 0.5, 1.0))
    C = 10 ** logC
    aof = C * (pr_psia ** 2) ** n
    return {"n": n, "C": float(C), "AOF_mscf_d": float(aof), "pr_psia": float(pr_psia)}

def arps_rate(qi, Di, b, t):
    t = np.asarray(t, dtype=float)
    if abs(b) < 1e-6:
        return qi * np.exp(-Di * t)
    return qi / (1 + b * Di * t) ** (1.0 / b)

def fit_arps_decline(rate_series):
    q = rate_series.dropna()
    q = q[q > 0]
    if len(q) < 4:
        return None
    t = np.arange(len(q), dtype=float)
    qvals = q.values
    qi0 = qvals[0]
    best = None
    for b in np.arange(0.0, 1.01, 0.1):
        for Di in np.arange(0.005, 0.301, 0.005):
            qhat = arps_rate(qi0, Di, b, t)
            sse = float(np.sum((qhat - qvals) ** 2))
            if best is None or sse < best["sse"]:
                best = {"qi": float(qi0), "Di": float(Di), "b": float(b), "sse": sse}
    return best

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

    if "Gas_Rate_mscf_d" in df:
        g = df["Gas_Rate_mscf_d"].dropna()
        if len(g) >= 2:
            a["gas_initial"] = float(g.iloc[0])
            a["gas_current"] = float(g.iloc[-1])
            a["gas_change_pct"] = float((g.iloc[-1] / g.iloc[0] - 1) * 100) if g.iloc[0] else 0
            positive = g[g > 0]
            if len(positive) >= 3:
                x = np.arange(len(positive), dtype=float)
                slope, intercept = np.polyfit(x, np.log(positive.values), 1)
                a["gas_decline_rate_monthly"] = float(-slope)
                a["gas_decline_rate_annual_pct"] = float((1 - math.exp(12 * slope)) * 100)
                a["gas_forecast_q12"] = float(math.exp(intercept + slope * (len(positive) - 1 + 12)))

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

    if "Condensate_Rate_bpd" in df:
        c = df["Condensate_Rate_bpd"].dropna()
        if len(c):
            a["condensate_current"] = float(c.iloc[-1])

    if "Condensate_Rate_bpd" in df and "Gas_Rate_mscf_d" in df:
        tmp = df[["Condensate_Rate_bpd", "Gas_Rate_mscf_d"]].dropna()
        if len(tmp):
            gas_mmscf = tmp["Gas_Rate_mscf_d"].iloc[-1] / 1000.0
            a["cgr_bbl_mmscf"] = float(tmp["Condensate_Rate_bpd"].iloc[-1] / gas_mmscf) if gas_mmscf else 0

    return a

def detect_anomalies(df):
    rows = []
    for col, label in [
        ("Oil_Rate_bpd", "Oil rate"),
        ("Pressure_psia", "Pressure"),
        ("Water_Rate_bpd", "Water rate"),
        ("Gas_Rate_mscf_d", "Gas rate"),
        ("Condensate_Rate_bpd", "Condensate rate"),
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

def gas_economics(a, assumptions, forecast_months):
    months = np.arange(1, forecast_months + 1)
    discount = 0.10 / 12
    total_revenue = np.zeros(forecast_months)
    detail = {}

    if a.get("oil_current", 0) > 0:
        D = a.get("decline_rate_monthly")
        if D is not None and D > 0:
            q = a["oil_current"] * np.exp(-D * months)
        else:
            q = np.full(forecast_months, a["oil_current"])
        monthly_bbl = q * 30.4375
        rev = monthly_bbl * assumptions["oil_price"]
        detail["oil_forecast_bbl"] = float(monthly_bbl.sum())
        detail["oil_revenue"] = float(rev.sum())
        total_revenue += rev

    if a.get("gas_current", 0) > 0:
        Dg = a.get("gas_decline_rate_monthly")
        if Dg is not None and Dg > 0:
            qg = a["gas_current"] * np.exp(-Dg * months)
        else:
            qg = np.full(forecast_months, a["gas_current"])
        monthly_mscf = qg * 30.4375
        shrink_factor = 1 - assumptions.get("shrinkage_pct", 0) / 100
        sellable_mscf = monthly_mscf * shrink_factor
        net_price = max(assumptions.get("gas_price", 0) - assumptions.get("processing_fee", 0), 0)
        rev = sellable_mscf * net_price
        detail["gas_forecast_mscf"] = float(monthly_mscf.sum())
        detail["gas_sellable_mscf"] = float(sellable_mscf.sum())
        detail["gas_revenue"] = float(rev.sum())
        total_revenue += rev

    if a.get("condensate_current", 0) > 0:
        cond_bbl = np.full(forecast_months, a["condensate_current"] * 30.4375)
        rev = cond_bbl * assumptions.get("condensate_price", 0)
        detail["condensate_forecast_bbl"] = float(cond_bbl.sum())
        detail["condensate_revenue"] = float(rev.sum())
        total_revenue += rev

    opex_total = assumptions["opex_monthly"] * forecast_months
    cash = total_revenue - assumptions["opex_monthly"]
    npv = -assumptions["capex"] + np.sum(cash / ((1 + discount) ** months))

    detail["opex"] = float(opex_total)
    detail["capex"] = float(assumptions["capex"])
    detail["total_revenue"] = float(total_revenue.sum())
    detail["npv"] = float(npv)
    return detail

def health_score(a, anomalies):
    score = 100
    primary_decline = a.get("oil_decline_pct")
    if primary_decline is None:
        annual_gas_decline = a.get("gas_decline_rate_annual_pct")
        primary_decline = max(annual_gas_decline, 0) / 2 if annual_gas_decline else 0
    score -= min(max(primary_decline, 0) * 0.7, 35)
    score -= min(max(a.get("pressure_decline_pct", 0), 0) * 0.35, 20)
    wc_increase = a.get("watercut_current_pct", 0) - a.get("watercut_initial_pct", 0)
    score -= min(max(wc_increase, 0) * 0.5, 25)
    score -= min(len(anomalies) * 5, 20)
    return max(0, min(100, round(score)))

def build_context(a, anomalies, econ, material_balance=None, aof=None):
    ctx = {
        "metrics": a,
        "anomalies": anomalies.to_dict("records"),
        "economics": econ,
    }
    if material_balance:
        ctx["material_balance_pz"] = material_balance
    if aof:
        ctx["deliverability_aof"] = aof
    return json.dumps(ctx, indent=2, default=str)

def ask_groq(question, context):
    try:
        api_key = st.secrets["GROQ_API_KEY"]
        api_key = str(api_key).strip()
        if not api_key:
            return "Groq API key is empty. Please check Streamlit Secrets."

        client = Groq(api_key=api_key)
        prompt = f"""
You are REX, a petroleum reservoir-engineering decision-support assistant covering both
oil and gas / gas-condensate reservoirs. Your scope includes production performance,
PVT properties (Z-factor, Bg, viscosity), P/Z volumetric material balance and OGIP,
deliverability / Absolute Open Flow (AOF) testing, Arps decline curve analysis
(exponential and hyperbolic), and screening-level economics.
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
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": "You are REX, an AI reservoir engineering assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=700,
        )
        return response.choices[0].message.content
    except KeyError:
        return "Groq API key is not configured. Add GROQ_API_KEY to your Streamlit Secrets."
    except Exception as e:
        return f"Groq request failed: {e}"

# ============================================================
# UI HEADER LAYOUT
# ============================================================
st.title("🛢️ REX | Reservoir Engineering eXpert")
st.markdown("### AI-Powered Oil & Gas Reservoir Decision Support Framework")
st.markdown("---")

st.session_state.setdefault("aof_result", None)

with st.sidebar:
    st.header("🗂️ Data Input")
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
    st.header("💰 Oil Economics")
    oil_price = st.number_input("Oil price ($/bbl)", min_value=0.0, value=70.0, step=5.0)
    opex_monthly = st.number_input("Monthly OPEX ($)", min_value=0.0, value=18000.0, step=1000.0)
    capex = st.number_input("CAPEX ($)", min_value=0.0, value=100000.0, step=10000.0)
    forecast_months = st.slider("Economic forecast (months)", 6, 60, 24)

    st.divider()
    st.header("🌡️ Gas Reservoir Parameters")
    sg_gas = st.number_input(
        "Gas specific gravity (air = 1.0)",
        min_value=0.55, max_value=1.20, value=0.65, step=0.01,
    )
    res_temp_F = st.number_input(
        "Reservoir temperature (°F)",
        min_value=60.0, max_value=400.0, value=180.0, step=5.0,
    )

    st.divider()
    st.header("📊 Gas Economics")
    gas_price = st.number_input("Gas price ($/mscf)", min_value=0.0, value=3.50, step=0.25)
    shrinkage_pct = st.number_input("Shrinkage / flare (%)", min_value=0.0, max_value=30.0, value=3.0, step=0.5)
    processing_fee = st.number_input("Processing fee ($/mscf)", min_value=0.0, value=0.35, step=0.05)
    condensate_price = st.number_input("Condensate price ($/bbl)", min_value=0.0, value=60.0, step=5.0)

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
            st.error("The file was uploaded successfully, but no recognized reservoir columns were found.")
            st.stop()
    except Exception as e:
        st.error(f"Could not read the uploaded file: {e}")
        st.stop()

elif use_demo:
    raw = DEMO_DATA.copy()
    df, mapping = prepare_data(raw)
    source_name = "Built-in Demo Dataset"

else:
    st.info("👈 Please upload a CSV/XLSX file or check 'Use built-in demo dataset' in the sidebar to begin.")
    st.stop()

if len(df) < 2:
    st.error("The dataset needs at least two usable rows.")
    st.stop()

a = analyze(df)
anomalies = detect_anomalies(df)

has_gas_data = "Gas_Rate_mscf_d" in df and df["Gas_Rate_mscf_d"].notna().sum() >= 2
has_pressure_data = "Pressure_psia" in df and df["Pressure_psia"].notna().sum() >= 2

pvt_df = compute_gas_pvt(df, sg_gas, res_temp_F) if (has_gas_data and has_pressure_data) else None
pz_df, mb_result = (
    pz_material_balance(df, sg_gas, res_temp_F) if (has_gas_data and has_pressure_data) else (None, None)
)

assumptions = {
    "oil_price": oil_price,
    "opex_monthly": opex_monthly,
    "capex": capex,
    "gas_price": gas_price,
    "shrinkage_pct": shrinkage_pct,
    "processing_fee": processing_fee,
    "condensate_price": condensate_price,
}
econ = gas_economics(a, assumptions, forecast_months)
score = health_score(a, anomalies)

st.success(f"Active Source: **{source_name}** • Loaded **{len(df):,}** records successfully.")

tabs = st.tabs([
    "🏠 Overview", "📊 Performance", "⛽ Gas PVT", "📐 Material Balance",
    "🎯 Deliverability", "🚨 Anomalies", "🧠 AI Diagnosis",
    "📉 Forecast", "💰 Economics", "💬 Ask REX"
])

with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reservoir Health Score", f"{score}/100")
    c2.metric("Current Oil Rate", f"{a.get('oil_current', 0):,.0f} bpd" if "oil_current" in a else "N/A")
    c3.metric("Current Gas Rate", f"{a.get('gas_current', 0):,.0f} mscf/d" if has_gas_data else "N/A")
    c4.metric("Current Pressure", f"{a.get('pressure_current', 0):,.0f} psia" if has_pressure_data else "N/A")

    st.markdown("<br>", unsafe_allow_html=True)
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Water Cut", f"{a.get('watercut_current_pct', 0):.1f}%" if "watercut_current_pct" in a else "N/A")
    c6.metric("GOR", f"{a.get('gor_current', 0):,.0f} scf/STB" if "gor_current" in a else "N/A")
    c7.metric("CGR", f"{a.get('cgr_bbl_mmscf', 0):,.1f} bbl/MMscf" if a.get("cgr_bbl_mmscf") else "N/A")
    c8.metric("OGIP (P/Z)", f"{mb_result['OGIP_Bscf']:.2f} Bscf" if mb_result and mb_result.get("OGIP_Bscf") else "N/A")

with tabs[1]:
    st.subheader("Production & Pressure Performance Trends")
    if "Oil_Rate_bpd" in df:
        st.plotly_chart(px.line(df, x="Date", y="Oil_Rate_bpd", markers=True, title="Oil Production Rate (bpd)"), use_container_width=True)
    if "Gas_Rate_mscf_d" in df:
        st.plotly_chart(px.line(df, x="Date", y="Gas_Rate_mscf_d", markers=True, title="Gas Production Rate (mscf/d)"), use_container_width=True)
    if "Pressure_psia" in df:
        st.plotly_chart(px.line(df, x="Date", y="Pressure_psia", markers=True, title="Reservoir Pressure (psia)"), use_container_width=True)

with tabs[2]:
    st.subheader("Gas PVT Property Modeling")
    if not has_gas_data or not has_pressure_data:
        st.info("Gas rate and pressure columns are required for PVT evaluation.")
    else:
        Tpc, Ppc = pseudocritical_properties(sg_gas)
        c1, c2 = st.columns(2)
        c1.metric("Pseudo-critical Tpc", f"{Tpc:,.1f} °R")
        c2.metric("Pseudo-critical Ppc", f"{Ppc:,.1f} psia")
        
        plot_df = df.copy()
        plot_df["Z"] = pvt_df["Z"]
        st.plotly_chart(px.line(plot_df, x="Date", y="Z", markers=True, title="Gas Z-factor Profile"), use_container_width=True)

with tabs[3]:
    st.subheader("P/Z Material Balance & OGIP")
    if mb_result:
        c1, c2, c3 = st.columns(3)
        c1.metric("Estimated OGIP", f"{mb_result['OGIP_Bscf']:,.2f} Bscf")
        c2.metric("Cumulative Gas", f"{mb_result['cum_gas_produced_MMscf']:,.0f} MMscf")
        c3.metric("Fit Quality (R²)", f"{mb_result['r_squared']:.3f}")
        fig = px.scatter(pz_df, x="Cum_Gas_MMscf", y="P_over_Z", title="P/Z vs Cumulative Production")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Insufficient data for Material Balance calculation.")

with tabs[4]:
    st.subheader("Gas Well Deliverability (AOF)")
    pr_test = st.number_input("Reservoir Pressure Pr (psia)", min_value=0.0, value=float(a.get("pressure_current", 3000) or 3000))
    default_test = pd.DataFrame({"Rate_mscf_d": [1000.0, 2000.0, 3000.0], "Pwf_psia": [2800.0, 2500.0, 2100.0]})
    test_data = st.data_editor(default_test, num_rows="dynamic", use_container_width=True)
    if st.button("Calculate AOF"):
        res = rawlins_schellhardt(test_data["Rate_mscf_d"], test_data["Pwf_psia"], pr_test)
        if res:
            st.success(f"Calculated Absolute Open Flow (AOF): **{res['AOF_mscf_d']:,.0f} mscf/d**")

with tabs[5]:
    st.subheader("Automatic Anomaly Detection")
    if anomalies.empty:
        st.success("No production anomalies detected above screening thresholds.")
    else:
        st.dataframe(anomalies, use_container_width=True, hide_index=True)

with tabs[6]:
    st.subheader("AI Reservoir Diagnosis")
    context = build_context(a, anomalies, econ, mb_result, st.session_state.get("aof_result"))
    if st.button("🔎 Run AI Diagnosis", type="primary"):
        with st.spinner("REX is analyzing operational trends..."):
            ans = ask_groq("Provide a concise engineering diagnosis of this well's performance.", context)
        st.markdown(ans)

with tabs[7]:
    st.subheader("Decline Curve Analysis & Forecasting")
    stream = st.radio("Stream", ["Oil", "Gas"], horizontal=True)
    st.info(f"Configure forecast parameters for {stream} production stream.")

with tabs[8]:
    st.subheader("Screening Economics Summary")
    cols = st.columns(4)
    cols[0].metric("Total Revenue", f"${econ['total_revenue']:,.0f}")
    cols[1].metric("Total OPEX", f"${econ['opex']:,.0f}")
    cols[2].metric("CAPEX", f"${econ['capex']:,.0f}")
    cols[3].metric("Project NPV (10%)", f"${econ['npv']:,.0f}")

with tabs[9]:
    st.subheader("💬 Ask REX Assistant")
    if "chat" not in st.session_state:
        st.session_state.chat = []
    for role, msg in st.session_state.chat:
        with st.chat_message(role):
            st.markdown(msg)
    question = st.chat_input("Ask REX a question about this reservoir...")
    if question:
        st.session_state.chat.append(("user", question))
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            resp = ask_groq(question, build_context(a, anomalies, econ, mb_result, st.session_state.get("aof_result")))
            st.markdown(resp)
            st.session_state.chat.append(("assistant", resp))
