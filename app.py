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
# (Standing pseudo-criticals, Dranchuk & Abou-Kassem Z-factor,
#  Lee-Gonzalez-Eakin viscosity, standard gas FVF)
# ============================================================
R_CONST = 10.732   # psia*ft3 / (lb-mol*R)
PSC = 14.696       # psia
TSC = 520.0        # deg R (60 F)

def pseudocritical_properties(sg_gas, is_condensate=False):
    """Standing (1977) correlations for pseudo-critical temperature (deg R)
    and pressure (psia) of a natural gas mixture from its specific gravity."""
    if is_condensate:
        Tpc = 187 + 330 * sg_gas - 71.5 * sg_gas ** 2
        Ppc = 706 - 51.7 * sg_gas - 11.1 * sg_gas ** 2
    else:
        Tpc = 168 + 325 * sg_gas - 12.5 * sg_gas ** 2
        Ppc = 677 + 15.0 * sg_gas - 37.5 * sg_gas ** 2
    return Tpc, Ppc

def z_factor_dak(Tpr, Ppr, tol=1e-8, max_iter=100):
    """Dranchuk & Abou-Kassem (1975) real-gas deviation factor, solved by
    bisection so no external optimization library is required."""
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
        return 0.90  # fallback if correlation range is exceeded
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
    """Gas formation volume factor Bg, reservoir ft3 per standard ft3."""
    if P_psia <= 0:
        return np.nan
    return 0.02827 * Z * T_R / P_psia

def gas_viscosity_cp(T_R, P_psia, Z, sg_gas):
    """Lee, Gonzalez & Eakin (1966) gas viscosity correlation, centipoise."""
    if Z <= 0 or T_R <= 0 or P_psia <= 0:
        return np.nan
    M = 28.97 * sg_gas
    rho_g = (P_psia * M) / (Z * R_CONST * T_R)  # lb/ft3
    K = ((9.4 + 0.02 * M) * T_R ** 1.5) / (209 + 19 * M + T_R)
    X = 3.5 + 986.0 / T_R + 0.01 * M
    Y = 2.4 - 0.2 * X
    return K * math.exp(X * (rho_g / 62.4) ** Y) * 1e-4

def compute_gas_pvt(df, sg_gas, res_temp_F):
    """Row-by-row Z, Bg, and viscosity from the Pressure column."""
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

# ============================================================
# OIL PVT CORRELATIONS
# (Standing (1947) bubble-point/Rs/Bo, Beggs & Robinson (1975) oil
#  viscosity, Vasquez & Beggs (1980) undersaturated compressibility &
#  viscosity correction -- this is the standard black-oil correlation
#  set compiled and recommended in McCain's "Properties of Petroleum
#  Fluids" for screening-level PVT work when no lab PVT report exists.)
# ============================================================
def oil_specific_gravity(api):
    """API gravity -> stock-tank oil specific gravity (water = 1.0)."""
    return 141.5 / (131.5 + api)

def standing_rs(p_psia, api, sg_gas, t_f):
    """Standing (1947) solution GOR, scf/STB, valid at/below the
    bubble-point pressure (saturated oil)."""
    p_psia = max(p_psia, 0.0)
    x = (p_psia / 18.2 + 1.4) * 10 ** (0.0125 * api - 0.00091 * t_f)
    return sg_gas * x ** 1.2048

def standing_bo(rs, api, sg_gas, t_f):
    """Standing (1947) saturated oil formation volume factor, rb/STB."""
    sg_oil = oil_specific_gravity(api)
    f = rs * (sg_gas / sg_oil) ** 0.5 + 1.25 * t_f
    return 0.9759 + 0.00012 * f ** 1.2

def vasquez_beggs_co(p_psia, rsb, api, sg_gas, t_f):
    """Vasquez & Beggs (1980) undersaturated oil isothermal
    compressibility, 1/psi, for use above the bubble point."""
    if p_psia <= 0:
        return np.nan
    return (-1433 + 5 * rsb + 17.2 * t_f - 1180 * sg_gas + 12.61 * api) / (1e5 * p_psia)

def beggs_robinson_dead_oil_visc(api, t_f):
    """Beggs & Robinson (1975) dead-oil viscosity, cp."""
    x = 10 ** (3.0324 - 0.02023 * api) * t_f ** (-1.163)
    return 10 ** x - 1

def beggs_robinson_live_oil_visc(mu_od, rs):
    """Beggs & Robinson (1975) saturated (live) oil viscosity, cp."""
    a = 10.715 * (rs + 100) ** -0.515
    b = 5.44 * (rs + 150) ** -0.338
    return a * mu_od ** b

def vasquez_beggs_visc_above_pb(mu_ob, p_psia, pb_psia):
    """Vasquez & Beggs (1980) undersaturated oil viscosity correction
    above the bubble point, cp."""
    if pb_psia <= 0:
        return mu_ob
    m = 2.6 * p_psia ** 1.187 * math.exp(-11.513 - 8.98e-5 * p_psia)
    return mu_ob * (p_psia / pb_psia) ** m

def compute_oil_pvt(df, api, sg_gas, t_f, pb_psia):
    """Row-by-row Rs, Bo, oil viscosity and compressibility from the
    Pressure column, honoring the saturated/undersaturated split at
    the user-supplied bubble-point pressure."""
    if "Pressure_psia" not in df:
        return None

    rsb = standing_rs(pb_psia, api, sg_gas, t_f)
    bob = standing_bo(rsb, api, sg_gas, t_f)
    mu_od = beggs_robinson_dead_oil_visc(api, t_f)
    mu_ob = beggs_robinson_live_oil_visc(mu_od, rsb)

    rows = []
    for _, p in df["Pressure_psia"].items():
        if pd.isna(p) or p <= 0:
            rows.append({"Rs_scf_stb": np.nan, "Bo_rb_stb": np.nan,
                         "Oil_Viscosity_cp": np.nan, "co_per_psi": np.nan,
                         "Regime": None})
            continue
        if p <= pb_psia:
            rs = standing_rs(p, api, sg_gas, t_f)
            bo = standing_bo(rs, api, sg_gas, t_f)
            mu_o = beggs_robinson_live_oil_visc(mu_od, rs)
            co = np.nan
            regime = "Saturated"
        else:
            rs = rsb
            co = vasquez_beggs_co(p, rsb, api, sg_gas, t_f)
            bo = bob * math.exp(-co * (p - pb_psia))
            mu_o = vasquez_beggs_visc_above_pb(mu_ob, p, pb_psia)
            regime = "Undersaturated"
        rows.append({"Rs_scf_stb": rs, "Bo_rb_stb": bo, "Oil_Viscosity_cp": mu_o,
                     "co_per_psi": co, "Regime": regime})

    pvt = pd.DataFrame(rows, index=df.index)
    pvt["Rsb_scf_stb"], pvt["Bob_rb_stb"], pvt["Pb_psia"] = rsb, bob, pb_psia
    pvt["Dead_Oil_Viscosity_cp"], pvt["Sg_Oil"] = mu_od, oil_specific_gravity(api)
    return pvt

# ============================================================
# BUCKLEY-LEVERETT FRACTIONAL FLOW (Corey relative permeability)
# ============================================================
def corey_relperm(sw, swc, sor, krw_max, kro_max, nw, no):
    """Corey-type normalized relative permeability curves."""
    sw = np.asarray(sw, dtype=float)
    denom = max(1.0 - swc - sor, 1e-9)
    s_star = np.clip((sw - swc) / denom, 0.0, 1.0)
    krw = krw_max * s_star ** nw
    kro = kro_max * (1.0 - s_star) ** no
    krw = np.where(sw <= swc, 0.0, krw)
    kro = np.where(sw >= (1.0 - sor), 0.0, kro)
    return krw, kro

def fractional_flow_curve(swc, sor, krw_max, kro_max, nw, no, mu_w, mu_o, n_points=201):
    """fw(Sw) for horizontal, incompressible, immiscible displacement
    (no gravity or capillary pressure terms): fw = 1 / (1 + (kro*muw)/(krw*muo))."""
    sw = np.linspace(swc, 1.0 - sor, n_points)
    krw, kro = corey_relperm(sw, swc, sor, krw_max, kro_max, nw, no)
    with np.errstate(divide="ignore", invalid="ignore"):
        fw = 1.0 / (1.0 + (kro * mu_w) / (krw * mu_o))
    fw = np.nan_to_num(fw, nan=0.0, posinf=1.0, neginf=0.0)
    fw = np.clip(fw, 0.0, 1.0)
    return sw, fw

def welge_shock_front(sw, fw, swc):
    """Welge (1952) tangent construction from (Swc, 0): finds the shock
    front saturation Swf and the average water saturation behind the
    front at breakthrough, Sw_bar, by maximizing fw / (Sw - Swc)."""
    denom = sw - swc
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(denom > 1e-9, fw / denom, 0.0)
    idx = int(np.argmax(slope))
    swf = float(sw[idx])
    fwf = float(fw[idx])
    tangent_slope = float(slope[idx])
    sw_bar_bt = swc + (1.0 / tangent_slope) if tangent_slope > 0 else np.nan
    return {"Swf": swf, "fwf": fwf, "tangent_slope": tangent_slope, "Sw_bar_breakthrough": sw_bar_bt}

# ============================================================
# P/Z MATERIAL BALANCE (volumetric OGIP estimate)
# ============================================================
def estimate_cum_gas(df):
    """Use a supplied cumulative-gas column if present; otherwise integrate
    the gas rate assuming each row is one month of production."""
    if "Cum_Gas_MMscf" in df and df["Cum_Gas_MMscf"].notna().sum() >= 2:
        return df["Cum_Gas_MMscf"]
    if "Gas_Rate_mscf_d" not in df:
        return None
    rate = df["Gas_Rate_mscf_d"].fillna(0)
    days_per_period = 30.4375
    incr_mmscf = rate * days_per_period / 1000.0
    return incr_mmscf.cumsum()

def estimate_cum_oil(df):
    """Integrate the oil rate assuming each row represents one month of
    production, giving cumulative oil in STB."""
    if "Oil_Rate_bpd" not in df:
        return None
    rate = df["Oil_Rate_bpd"].fillna(0)
    days_per_period = 30.4375
    return (rate * days_per_period).cumsum()

def pz_material_balance(df, sg_gas, res_temp_F):
    """Fit P/Z vs cumulative gas production; OGIP is the x-intercept."""
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

# ============================================================
# GAS WELL DELIVERABILITY (Rawlins-Schellhardt back-pressure test)
# ============================================================
def rawlins_schellhardt(rates_mscfd, pwf_psia, pr_psia):
    """Fit qg = C * (Pr^2 - Pwf^2)^n from multi-point test data and
    compute Absolute Open Flow (AOF) at Pwf = 0."""
    rates = np.asarray(rates_mscfd, dtype=float)
    pwf = np.asarray(pwf_psia, dtype=float)
    delta_p2 = pr_psia ** 2 - pwf ** 2
    mask = (delta_p2 > 0) & (rates > 0) & np.isfinite(delta_p2) & np.isfinite(rates)
    if mask.sum() < 2:
        return None
    x = np.log10(delta_p2[mask])
    y = np.log10(rates[mask])
    n, logC = np.polyfit(x, y, 1)
    n = float(np.clip(n, 0.5, 1.0))  # physically bounded: 0.5 (turbulent) - 1.0 (Darcy)
    C = 10 ** logC
    aof = C * (pr_psia ** 2) ** n
    return {"n": n, "C": float(C), "AOF_mscf_d": float(aof), "pr_psia": float(pr_psia)}

# ============================================================
# ARPS DECLINE CURVE (exponential is b=0; hyperbolic fits qi, Di, b)
# ============================================================
def arps_rate(qi, Di, b, t):
    t = np.asarray(t, dtype=float)
    if abs(b) < 1e-6:
        return qi * np.exp(-Di * t)
    return qi / (1 + b * Di * t) ** (1.0 / b)

def fit_arps_decline(rate_series):
    """Coarse grid-search fit of the Arps hyperbolic decline (no SciPy
    dependency). Good enough for screening-level EUR/forecast work."""
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
    """Combined oil + gas + condensate screening economics.

    assumptions keys: oil_price, opex_monthly, capex, gas_price,
    shrinkage_pct, processing_fee, condensate_price
    """
    months = np.arange(1, forecast_months + 1)
    discount = 0.10 / 12
    total_revenue = np.zeros(forecast_months)
    detail = {}

    # --- Oil stream ---
    if a.get("oil_current", 0) > 0:
        D = a.get("decline_rate_monthly")
        if D is not None and D > 0:
            q = a["oil_current"] * np.exp(-D * months)
        else:
            q = np.full(forecast_months, a["oil_current"])  # flat if stable/increasing
        monthly_bbl = q * 30.4375
        rev = monthly_bbl * assumptions["oil_price"]
        detail["oil_forecast_bbl"] = float(monthly_bbl.sum())
        detail["oil_revenue"] = float(rev.sum())
        total_revenue += rev

    # --- Gas stream ---
    if a.get("gas_current", 0) > 0:
        Dg = a.get("gas_decline_rate_monthly")
        if Dg is not None and Dg > 0:
            qg = a["gas_current"] * np.exp(-Dg * months)
        else:
            qg = np.full(forecast_months, a["gas_current"])  # flat if stable/increasing
        monthly_mscf = qg * 30.4375
        shrink_factor = 1 - assumptions.get("shrinkage_pct", 0) / 100
        sellable_mscf = monthly_mscf * shrink_factor
        net_price = max(assumptions.get("gas_price", 0) - assumptions.get("processing_fee", 0), 0)
        rev = sellable_mscf * net_price
        detail["gas_forecast_mscf"] = float(monthly_mscf.sum())
        detail["gas_sellable_mscf"] = float(sellable_mscf.sum())
        detail["gas_revenue"] = float(rev.sum())
        total_revenue += rev

    # --- Condensate stream (held flat at current rate - no separate decline fit) ---
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
        # Gas-only well: derive a dampened equivalent from annualized gas decline
        annual_gas_decline = a.get("gas_decline_rate_annual_pct")
        primary_decline = max(annual_gas_decline, 0) / 2 if annual_gas_decline else 0
    score -= min(max(primary_decline, 0) * 0.7, 35)
    score -= min(max(a.get("pressure_decline_pct", 0), 0) * 0.35, 20)
    wc_increase = a.get("watercut_current_pct", 0) - a.get("watercut_initial_pct", 0)
    score -= min(max(wc_increase, 0) * 0.5, 25)
    score -= min(len(anomalies) * 5, 20)
    return max(0, min(100, round(score)))

def build_context(a, anomalies, econ, material_balance=None, aof=None, oil_pvt_summary=None, fractional_flow=None):
    ctx = {
        "metrics": a,
        "anomalies": anomalies.to_dict("records"),
        "economics": econ,
    }
    if material_balance:
        ctx["material_balance_pz"] = material_balance
    if aof:
        ctx["deliverability_aof"] = aof
    if oil_pvt_summary:
        ctx["oil_pvt_mccain"] = oil_pvt_summary
    if fractional_flow:
        ctx["fractional_flow"] = fractional_flow
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
You are REX, a petroleum reservoir-engineering decision-support assistant covering both
oil and gas / gas-condensate reservoirs. Your scope includes production performance,
gas PVT properties (Z-factor, Bg, viscosity), oil PVT properties (Rs, Bo, oil viscosity,
compressibility via the Standing / Beggs-Robinson / Vasquez-Beggs correlations compiled
in McCain's Properties of Petroleum Fluids), Buckley-Leverett fractional flow and Welge
shock-front analysis, P/Z volumetric material balance and OGIP, deliverability /
Absolute Open Flow (AOF) testing, Arps decline curve analysis (exponential and
hyperbolic), and screening-level economics.
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
st.caption("Reservoir Engineering eXpert • AI-powered oil & gas reservoir decision support")

st.session_state.setdefault("aof_result", None)

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
    st.header("Oil economics")
    oil_price = st.number_input("Oil price ($/bbl)", min_value=0.0, value=70.0, step=5.0)
    opex_monthly = st.number_input("Monthly OPEX ($)", min_value=0.0, value=18000.0, step=1000.0)
    capex = st.number_input("CAPEX ($)", min_value=0.0, value=100000.0, step=10000.0)
    forecast_months = st.slider("Economic forecast (months)", 6, 60, 24)

    st.divider()
    st.header("Gas reservoir parameters")
    sg_gas = st.number_input(
        "Gas specific gravity (air = 1.0)",
        min_value=0.55, max_value=1.20, value=0.65, step=0.01,
        help="Used for Standing pseudo-critical properties and PVT correlations.",
    )
    res_temp_F = st.number_input(
        "Reservoir temperature (°F)",
        min_value=60.0, max_value=400.0, value=180.0, step=5.0,
    )

    st.divider()
    st.header("Oil PVT parameters (McCain)")
    oil_api = st.number_input(
        "Stock-tank oil gravity (°API)",
        min_value=5.0, max_value=70.0, value=35.0, step=1.0,
        help="Used for Standing (Rs, Bo) and Beggs-Robinson (viscosity) black-oil correlations.",
    )
    pb_psia = st.number_input(
        "Bubble-point pressure, Pb (psia)",
        min_value=0.0, value=2500.0, step=50.0,
        help="Pressure below which the oil is saturated (gas comes out of solution).",
    )

    st.divider()
    st.header("Fractional flow (Buckley-Leverett)")
    swc = st.number_input("Connate water saturation, Swc", min_value=0.0, max_value=0.6, value=0.20, step=0.01)
    sor = st.number_input("Residual oil saturation, Sor", min_value=0.0, max_value=0.6, value=0.25, step=0.01)
    krw_max = st.number_input("Endpoint krw @ Sor", min_value=0.01, max_value=1.0, value=0.35, step=0.01)
    kro_max = st.number_input("Endpoint kro @ Swc", min_value=0.01, max_value=1.0, value=0.85, step=0.01)
    nw = st.number_input("Corey water exponent, nw", min_value=1.0, max_value=6.0, value=2.5, step=0.1)
    no = st.number_input("Corey oil exponent, no", min_value=1.0, max_value=6.0, value=2.0, step=0.1)
    water_visc_cp = st.number_input("Water viscosity, μw (cp)", min_value=0.1, value=0.5, step=0.05)
    oil_visc_override_cp = st.number_input(
        "Oil viscosity for fractional flow, μo (cp)",
        min_value=0.1, value=2.0, step=0.1,
        help="Defaults to a manual value; if Oil PVT is available REX will offer to use the "
             "correlation-derived value at reservoir pressure instead.",
    )

    st.divider()
    st.header("Gas economics")
    gas_price = st.number_input("Gas price ($/mscf)", min_value=0.0, value=3.50, step=0.25)
    shrinkage_pct = st.number_input(
        "Shrinkage / plant fuel & flare (%)", min_value=0.0, max_value=30.0, value=3.0, step=0.5
    )
    processing_fee = st.number_input(
        "Gathering & processing fee ($/mscf)", min_value=0.0, value=0.35, step=0.05
    )
    condensate_price = st.number_input(
        "Condensate / NGL price ($/bbl)", min_value=0.0, value=60.0, step=5.0
    )

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
                "Gas Rate, Water Rate, Pressure, Cumulative Gas, and/or Condensate Rate."
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

# Gas PVT / material balance are only meaningful if the file actually contains
# gas production data (a Pressure column alone isn't enough to call something
# a "gas field" — it could just as easily be an oil well's reservoir pressure).
has_gas_data = "Gas_Rate_mscf_d" in df and df["Gas_Rate_mscf_d"].notna().sum() >= 2
has_pressure_data = "Pressure_psia" in df and df["Pressure_psia"].notna().sum() >= 2

pvt_df = compute_gas_pvt(df, sg_gas, res_temp_F) if (has_gas_data and has_pressure_data) else None
pz_df, mb_result = (
    pz_material_balance(df, sg_gas, res_temp_F) if (has_gas_data and has_pressure_data) else (None, None)
)

has_oil_data = "Oil_Rate_bpd" in df and df["Oil_Rate_bpd"].notna().sum() >= 2
oil_pvt_df = (
    compute_oil_pvt(df, oil_api, sg_gas, res_temp_F, pb_psia)
    if (has_oil_data and has_pressure_data) else None
)

oil_pvt_summary = None
if oil_pvt_df is not None:
    latest_visc = oil_pvt_df["Oil_Viscosity_cp"].dropna()
    oil_pvt_summary = {
        "Rsb_scf_stb": float(oil_pvt_df["Rsb_scf_stb"].iloc[0]),
        "Bob_rb_stb": float(oil_pvt_df["Bob_rb_stb"].iloc[0]),
        "Pb_psia": float(pb_psia),
        "latest_oil_viscosity_cp": float(latest_visc.iloc[-1]) if len(latest_visc) else None,
        "correlations": "Standing (1947) Rs/Bo; Beggs-Robinson (1975) viscosity; "
                        "Vasquez-Beggs (1980) undersaturated co and viscosity correction",
    }

fw_sw, fw_curve = fractional_flow_curve(swc, sor, krw_max, kro_max, nw, no, water_visc_cp, oil_visc_override_cp)
fractional_flow_summary = welge_shock_front(fw_sw, fw_curve, swc)
fractional_flow_summary.update({"Swc": swc, "Sor": sor, "mu_w_cp": water_visc_cp, "mu_o_cp": oil_visc_override_cp})

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
    "🏠 Overview", "📊 Performance", "⛽ Gas PVT", "🛢️ Oil PVT (McCain)",
    "🌊 Fractional Flow", "📐 Material Balance",
    "🎯 Deliverability", "🚨 Anomalies", "🧠 AI Diagnosis",
    "📉 Forecast", "💰 Economics", "💬 Ask REX"
])

with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reservoir Health", f"{score}/100")
    c2.metric("Current Oil Rate", f"{a.get('oil_current', 0):,.0f} bpd" if "oil_current" in a else "N/A")
    c3.metric("Current Gas Rate", f"{a.get('gas_current', 0):,.0f} mscf/d" if has_gas_data else "N/A")
    c4.metric("Pressure", f"{a.get('pressure_current', 0):,.0f} psia" if has_pressure_data else "N/A")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Water Cut", f"{a.get('watercut_current_pct', 0):.1f}%" if "watercut_current_pct" in a else "N/A")
    c6.metric("GOR", f"{a.get('gor_current', 0):,.0f} scf/STB" if "gor_current" in a else "N/A")
    c7.metric("CGR", f"{a.get('cgr_bbl_mmscf', 0):,.1f} bbl/MMscf" if a.get("cgr_bbl_mmscf") else "N/A")
    c8.metric(
        "OGIP (P/Z)",
        f"{mb_result['OGIP_Bscf']:.2f} Bscf" if mb_result and mb_result.get("OGIP_Bscf") else "N/A",
    )

    st.subheader("Key engineering indicators")
    indicators = pd.DataFrame({
        "Indicator": ["Oil decline", "Gas rate change", "Pressure decline",
                      "Water-cut increase", "Detected anomalies"],
        "Value": [
            f"{a.get('oil_decline_pct', 0):.1f}%" if "oil_decline_pct" in a else "N/A",
            f"{a.get('gas_change_pct', 0):.1f}%" if "gas_change_pct" in a else "N/A",
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

    if "Gas_Rate_mscf_d" in df:
        st.plotly_chart(
            px.line(df, x="Date", y="Gas_Rate_mscf_d", markers=True, title="Gas Rate"),
            use_container_width=True,
        )

    if "Condensate_Rate_bpd" in df:
        st.plotly_chart(
            px.line(df, x="Date", y="Condensate_Rate_bpd", markers=True, title="Condensate Rate"),
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

    st.subheader("Analytics")

    if "Oil_Rate_bpd" in df and "Water_Rate_bpd" in df:
        cum_oil = estimate_cum_oil(df)
        chart = df.copy()
        chart["Cum_Oil_STB"] = cum_oil
        denominator = chart["Oil_Rate_bpd"] + chart["Water_Rate_bpd"]
        chart["Water_Cut_%"] = 100 * chart["Water_Rate_bpd"] / denominator.replace(0, np.nan)
        chart = chart.dropna(subset=["Cum_Oil_STB", "Water_Cut_%"])
        st.plotly_chart(
            px.line(chart, x="Cum_Oil_STB", y="Water_Cut_%", markers=True,
                    title="Water Cut % vs Cumulative Oil Production"),
            use_container_width=True,
        )
        st.caption(
            "Cumulative oil is estimated by integrating the oil rate assuming each row is one "
            "month of production (use a real cumulative-oil column in your file for a more "
            "precise curve). A rising, accelerating water-cut trend against cumulative oil is a "
            "classic signature of water breakthrough / coning — cross-check against the "
            "Fractional Flow tab's shock-front saturation."
        )
    else:
        st.info("Oil rate and water rate data are both required for the water-cut vs cumulative-oil chart.")

with tabs[2]:
    st.subheader("Gas PVT Properties")

    if not has_gas_data:
        st.info(
            "No gas production column was detected in this dataset (REX looks for something like "
            "'Gas_Rate_mscf_d', 'Gas Rate', 'qg', etc.). Gas PVT properties only apply to a "
            "gas-bearing stream, so this section stays inactive until a gas rate column is present "
            "— check the 'Column Mapping' expander above to see what REX picked up from your file."
        )
    elif not has_pressure_data:
        st.info(
            "Gas rate data was found, but a Pressure column is also required to compute "
            "pressure-dependent PVT properties (Z-factor, Bg, viscosity)."
        )
    else:
        Tpc, Ppc = pseudocritical_properties(sg_gas)
        latest_z = pvt_df["Z"].dropna()
        c1, c2, c3 = st.columns(3)
        c1.metric("Pseudo-critical Tpc", f"{Tpc:,.1f} °R")
        c2.metric("Pseudo-critical Ppc", f"{Ppc:,.1f} psia")
        c3.metric("Latest Z-factor", f"{latest_z.iloc[-1]:.3f}" if len(latest_z) else "N/A")

        plot_df = df.copy()
        plot_df["Z"] = pvt_df["Z"]
        plot_df["Bg_rcf_scf"] = pvt_df["Bg_rcf_scf"]
        plot_df["Gas_Viscosity_cp"] = pvt_df["Gas_Viscosity_cp"]

        st.plotly_chart(
            px.line(plot_df, x="Date", y="Z", markers=True,
                    title="Gas Z-factor (Dranchuk & Abou-Kassem)"),
            use_container_width=True,
        )
        st.plotly_chart(
            px.line(plot_df, x="Date", y="Bg_rcf_scf", markers=True,
                    title="Gas Formation Volume Factor (Bg)"),
            use_container_width=True,
        )
        st.plotly_chart(
            px.line(plot_df, x="Date", y="Gas_Viscosity_cp", markers=True,
                    title="Gas Viscosity"),
            use_container_width=True,
        )

        if "Condensate_Rate_bpd" in df and "Gas_Rate_mscf_d" in df:
            cgr_df = df.copy()
            cgr_df["CGR_bbl_MMscf"] = cgr_df["Condensate_Rate_bpd"] / (
                cgr_df["Gas_Rate_mscf_d"] / 1000
            ).replace(0, np.nan)
            st.plotly_chart(
                px.line(cgr_df, x="Date", y="CGR_bbl_MMscf", markers=True,
                        title="Condensate-Gas Ratio (CGR)"),
                use_container_width=True,
            )

        st.caption(
            "Z-factor via Dranchuk & Abou-Kassem (1975); pseudo-criticals via Standing (1977); "
            "viscosity via Lee, Gonzalez & Eakin (1966). Screening-level correlations for sweet, "
            "non-associated natural gas — apply sour-gas corrections (Wichert-Aziz) separately if "
            "H2S/CO2 content is significant. Note: Tpc and Ppc depend only on the gas specific "
            "gravity you set in the sidebar, not on the uploaded file — they'll only change if you "
            "adjust that value, since gas composition isn't something REX can infer from rate data."
        )

with tabs[3]:
    st.subheader("Oil PVT Properties (Black-Oil Correlations)")

    if not has_oil_data:
        st.info(
            "No oil production column was detected in this dataset (REX looks for something like "
            "'Oil_Rate_bpd', 'Oil Rate', 'qo', etc.). Oil PVT properties only apply to a liquid "
            "stream, so this section stays inactive until an oil rate column is present."
        )
    elif not has_pressure_data:
        st.info(
            "Oil rate data was found, but a Pressure column is also required to compute "
            "pressure-dependent oil PVT properties (Rs, Bo, viscosity)."
        )
    else:
        rsb = oil_pvt_df["Rsb_scf_stb"].iloc[0]
        bob = oil_pvt_df["Bob_rb_stb"].iloc[0]
        mu_od = oil_pvt_df["Dead_Oil_Viscosity_cp"].iloc[0]
        sg_oil = oil_pvt_df["Sg_Oil"].iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Stock-tank oil SG", f"{sg_oil:.3f}")
        c2.metric("Rs @ Pb", f"{rsb:,.0f} scf/STB")
        c3.metric("Bo @ Pb", f"{bob:.3f} rb/STB")
        c4.metric("Dead-oil viscosity", f"{mu_od:.2f} cp")

        plot_df = df.copy()
        plot_df["Rs_scf_stb"] = oil_pvt_df["Rs_scf_stb"]
        plot_df["Bo_rb_stb"] = oil_pvt_df["Bo_rb_stb"]
        plot_df["Oil_Viscosity_cp"] = oil_pvt_df["Oil_Viscosity_cp"]

        st.plotly_chart(
            px.line(plot_df, x="Date", y="Rs_scf_stb", markers=True,
                    title="Solution GOR, Rs (Standing, 1947)"),
            use_container_width=True,
        )
        st.plotly_chart(
            px.line(plot_df, x="Date", y="Bo_rb_stb", markers=True,
                    title="Oil Formation Volume Factor, Bo (Standing, 1947)"),
            use_container_width=True,
        )
        st.plotly_chart(
            px.line(plot_df, x="Date", y="Oil_Viscosity_cp", markers=True,
                    title="Oil Viscosity (Beggs & Robinson, 1975)"),
            use_container_width=True,
        )

        n_undersat = int((oil_pvt_df["Regime"] == "Undersaturated").sum())
        n_sat = int((oil_pvt_df["Regime"] == "Saturated").sum())
        st.caption(
            f"Regime split across the dataset: {n_sat} saturated point(s) at/below Pb, "
            f"{n_undersat} undersaturated point(s) above Pb ({pb_psia:,.0f} psia). "
            "Correlations: Standing (1947) for Rs and saturated Bo, Beggs & Robinson (1975) for "
            "dead- and live-oil viscosity, Vasquez & Beggs (1980) for undersaturated compressibility "
            "and the viscosity correction above Pb — this is the standard black-oil correlation set "
            "compiled and recommended in McCain's *Properties of Petroleum Fluids* for screening "
            "PVT work when no lab report is available. Always validate against a measured PVT "
            "report (differential liberation / CCE) before using these in a full reservoir study."
        )
        st.dataframe(oil_pvt_df, use_container_width=True, hide_index=True)

with tabs[4]:
    st.subheader("Buckley-Leverett Fractional Flow")
    st.caption(
        "Corey-type relative permeability curves feed the fractional flow equation "
        "fw(Sw) = 1 / [1 + (kro·μw)/(krw·μo)] for a horizontal, incompressible waterflood "
        "(no gravity or capillary-pressure terms)."
    )

    mu_o_source = oil_visc_override_cp
    if oil_pvt_df is not None:
        latest_mu_o = oil_pvt_df["Oil_Viscosity_cp"].dropna()
        if len(latest_mu_o):
            use_pvt_visc = st.checkbox(
                f"Use correlation-derived oil viscosity at the latest reservoir pressure "
                f"({latest_mu_o.iloc[-1]:.2f} cp) instead of the sidebar value",
                value=True,
            )
            if use_pvt_visc:
                mu_o_source = float(latest_mu_o.iloc[-1])

    sw, fw = fractional_flow_curve(swc, sor, krw_max, kro_max, nw, no, water_visc_cp, mu_o_source)
    shock = welge_shock_front(sw, fw, swc)

    c1, c2, c3 = st.columns(3)
    c1.metric("Shock-front saturation, Swf", f"{shock['Swf']:.3f}")
    c2.metric("fw at shock front", f"{shock['fwf']:.3f}")
    c3.metric(
        "Avg Sw behind front @ breakthrough",
        f"{shock['Sw_bar_breakthrough']:.3f}" if shock["Sw_bar_breakthrough"] == shock["Sw_bar_breakthrough"] else "N/A",
    )

    krw, kro = corey_relperm(sw, swc, sor, krw_max, kro_max, nw, no)
    relperm_df = pd.DataFrame({"Sw": sw, "krw": krw, "kro": kro})
    fw_df = pd.DataFrame({"Sw": sw, "fw": fw})

    fig_relperm = px.line(relperm_df, x="Sw", y=["krw", "kro"], title="Corey Relative Permeability")
    st.plotly_chart(fig_relperm, use_container_width=True)

    fig_fw = px.line(fw_df, x="Sw", y="fw", title="Fractional Flow Curve, fw vs Sw")
    if shock["Sw_bar_breakthrough"] == shock["Sw_bar_breakthrough"]:
        fig_fw.add_scatter(
            x=[swc, shock["Sw_bar_breakthrough"]], y=[0, 1],
            mode="lines", name="Welge tangent (shock front)",
        )
        fig_fw.add_scatter(
            x=[shock["Swf"]], y=[shock["fwf"]], mode="markers",
            marker=dict(size=10, symbol="x"), name="Swf",
        )
    st.plotly_chart(fig_fw, use_container_width=True)

    st.caption(
        "Welge (1952) tangent construction from (Swc, 0) locates the saturation shock front Swf "
        "and the average water saturation behind the front at breakthrough. This is a 1-D, "
        "displacement-only screening tool — it ignores gravity segregation, capillary pressure, "
        "and reservoir heterogeneity, so treat results as indicative rather than a substitute for "
        "full numerical simulation."
    )

with tabs[5]:
    st.subheader("P/Z Material Balance (OGIP Estimate)")

    if not has_gas_data:
        st.info(
            "No gas production column was detected in this dataset. P/Z material balance is a "
            "gas-reservoir method and needs a gas rate (or cumulative gas) column to run — check "
            "the 'Column Mapping' expander above to confirm what REX read from your file."
        )
    elif pz_df is None or len(pz_df) < 3:
        st.info(
            "Material balance requires pressure data (≥3 usable points) in addition to the gas "
            "rate/cumulative-gas data. If cumulative gas isn't in your file, REX estimates it from "
            "the gas rate assuming each row is one month of production."
        )
    else:
        if mb_result:
            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Estimated OGIP",
                f"{mb_result['OGIP_Bscf']:,.2f} Bscf" if mb_result["OGIP_Bscf"] else "N/A",
            )
            c2.metric("Cum. Gas Produced", f"{mb_result['cum_gas_produced_MMscf']:,.0f} MMscf")
            c3.metric("R² (P/Z linear fit)", f"{mb_result['r_squared']:.3f}")

            if mb_result["recovery_factor_pct"] is not None:
                st.metric("Current Recovery Factor", f"{mb_result['recovery_factor_pct']:.1f}%")

            fig = px.scatter(pz_df, x="Cum_Gas_MMscf", y="P_over_Z",
                              title="P/Z vs Cumulative Gas Production")
            if mb_result["OGIP_MMscf"]:
                x_line = np.array([0, mb_result["OGIP_MMscf"]])
                y_line = mb_result["slope"] * x_line + mb_result["intercept"]
                fig.add_scatter(x=x_line, y=y_line, mode="lines", name="Linear trend (extrapolated)")
            st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Volumetric (P/Z) material balance assumes a closed, volumetric gas reservoir with "
            "negligible water influx or pore-volume compaction drive. OGIP is the x-intercept of "
            "the P/Z trend (where P/Z = 0). Treat this as a screening estimate — validate against "
            "independent volumetric and well-test methods, and watch for a non-linear P/Z trend, "
            "which usually signals water drive or abnormal pressure behavior."
        )
        st.dataframe(pz_df, use_container_width=True, hide_index=True)

with tabs[6]:
    st.subheader("Gas Well Deliverability (Rawlins-Schellhardt)")
    st.caption(
        "Enter multi-point (isochronal / flow-after-flow) test data to fit the empirical "
        "back-pressure equation qg = C·(Pr² − Pwf²)ⁿ and estimate Absolute Open Flow (AOF)."
    )

    pr_test = st.number_input(
        "Average reservoir / static pressure, Pr (psia)",
        min_value=0.0, value=float(a.get("pressure_current", 3000) or 3000), step=10.0,
    )

    default_test = pd.DataFrame({
        "Rate_mscf_d": [1000.0, 2000.0, 3000.0, 4000.0],
        "Pwf_psia": [2800.0, 2600.0, 2350.0, 2050.0],
    })
    test_data = st.data_editor(
        default_test, num_rows="dynamic", use_container_width=True, key="aof_editor"
    )

    if st.button("Calculate AOF"):
        result = rawlins_schellhardt(test_data["Rate_mscf_d"], test_data["Pwf_psia"], pr_test)
        if result is None:
            st.error("Need at least two valid (rate, Pwf) points with Pwf < Pr.")
        else:
            st.session_state["aof_result"] = result
            c1, c2, c3 = st.columns(3)
            c1.metric("Deliverability exponent (n)", f"{result['n']:.3f}")
            c2.metric("Performance coefficient (C)", f"{result['C']:.4g}")
            c3.metric("AOF", f"{result['AOF_mscf_d']:,.0f} mscf/d")

            pwf_range = np.linspace(0, pr_test, 50)
            q_range = result["C"] * (pr_test ** 2 - pwf_range ** 2) ** result["n"]
            ipr_df = pd.DataFrame({"Pwf_psia": pwf_range, "Rate_mscf_d": q_range})
            st.plotly_chart(
                px.line(ipr_df, x="Rate_mscf_d", y="Pwf_psia", title="Gas Well IPR (Deliverability Curve)"),
                use_container_width=True,
            )
            st.caption(
                "n typically ranges from 0.5 (fully turbulent flow) to 1.0 (fully laminar/Darcy flow). "
                "AOF is the theoretical maximum rate at Pwf = 0 psia and is commonly used for regulatory "
                "allowables — it is not a recommended sustained operating rate."
            )
    else:
        st.info("Add test points above and click 'Calculate AOF'.")

with tabs[7]:
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

with tabs[8]:
    st.subheader("AI Reservoir Diagnosis")
    context = build_context(a, anomalies, econ, mb_result, st.session_state.get("aof_result"), oil_pvt_summary, fractional_flow_summary)

    if st.button("🔎 Generate Engineering Diagnosis", type="primary"):
        with st.spinner("REX is interpreting the calculated indicators..."):
            answer = ask_groq(
                "Provide a concise reservoir performance diagnosis covering both liquid and gas "
                "behavior where data is available. Structure it as Observations, Possible Causes, "
                "Evidence, and Recommended Investigation.",
                context,
            )
        st.markdown(answer)
    else:
        st.info("Click the button to generate an AI-assisted interpretation of the calculated results.")

with tabs[9]:
    st.subheader("Production Forecast")

    stream = st.radio("Forecast stream", ["Oil", "Gas"], horizontal=True)
    rate_col = "Oil_Rate_bpd" if stream == "Oil" else "Gas_Rate_mscf_d"
    unit = "bpd" if stream == "Oil" else "mscf/d"

    if rate_col not in df:
        st.info(f"{stream} rate data is required for forecasting.")
    else:
        positive = df[rate_col].dropna()
        positive = positive[positive > 0]

        if len(positive) >= 4:
            model = st.radio("Decline model", ["Exponential", "Hyperbolic (Arps)"], horizontal=True)
            horizon = st.slider("Forecast horizon (months)", 6, 60, 24)
            t_hist = np.arange(len(positive), dtype=float)
            t_future = np.arange(len(positive) + horizon, dtype=float)

            if model == "Exponential":
                slope, intercept = np.polyfit(t_hist, np.log(positive.values), 1)
                fitted = np.exp(intercept + slope * t_future)
                decline_label = f"{-slope * 100:.2f}% per month (exponential)"
            else:
                fit = fit_arps_decline(positive)
                if fit is None:
                    st.info("Not enough positive rate points to fit a hyperbolic decline.")
                    fit = {"qi": positive.iloc[0], "Di": 0.0, "b": 0.0}
                fitted = arps_rate(fit["qi"], fit["Di"], fit["b"], t_future)
                decline_label = f"Di={fit['Di'] * 100:.2f}%/mo, b={fit['b']:.2f} (hyperbolic)"

            hist = pd.DataFrame({"Period": t_hist, "Rate": positive.values, "Type": "Historical"})
            fc = pd.DataFrame({
                "Period": t_future[len(positive):],
                "Rate": fitted[len(positive):],
                "Type": "Forecast",
            })
            plot_df = pd.concat([hist, fc], ignore_index=True)

            st.plotly_chart(
                px.line(plot_df, x="Period", y="Rate", color="Type", markers=True,
                        title=f"{stream} Rate — Historical + {model} Forecast ({unit})"),
                use_container_width=True,
            )
            c1, c2 = st.columns(2)
            c1.metric("Decline parameters", decline_label)
            c2.metric(f"Forecast {stream.lower()} rate at horizon", f"{fc['Rate'].iloc[-1]:,.0f} {unit}")
        else:
            st.info(f"At least 4 positive {stream.lower()} rate points are needed to fit a decline model.")

with tabs[10]:
    st.subheader("Production Economics (Oil + Gas + Condensate)")

    cols = st.columns(4)
    cols[0].metric("Total Revenue", f"${econ['total_revenue']:,.0f}")
    cols[1].metric("OPEX", f"${econ['opex']:,.0f}")
    cols[2].metric("CAPEX", f"${econ['capex']:,.0f}")
    cols[3].metric("NPV (10% disc.)", f"${econ['npv']:,.0f}")

    st.markdown("**Revenue breakdown**")
    breakdown_rows = []
    if "oil_revenue" in econ:
        breakdown_rows.append({
            "Stream": "Oil",
            "Forecast Volume": f"{econ['oil_forecast_bbl']:,.0f} bbl",
            "Revenue": f"${econ['oil_revenue']:,.0f}",
        })
    if "gas_revenue" in econ:
        breakdown_rows.append({
            "Stream": "Gas",
            "Forecast Volume": f"{econ['gas_sellable_mscf']:,.0f} mscf (sellable, after shrinkage)",
            "Revenue": f"${econ['gas_revenue']:,.0f}",
        })
    if "condensate_revenue" in econ:
        breakdown_rows.append({
            "Stream": "Condensate",
            "Forecast Volume": f"{econ['condensate_forecast_bbl']:,.0f} bbl",
            "Revenue": f"${econ['condensate_revenue']:,.0f}",
        })

    if breakdown_rows:
        st.dataframe(pd.DataFrame(breakdown_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No stream has enough rate data yet for an economics forecast.")

    st.caption(
        "Illustrative screening economics: oil/gas forecast via exponential decline on the current "
        "rate (held flat if the trend is not declining); condensate held flat at the current rate. "
        "Gas revenue nets shrinkage and gathering/processing fees before applying the gas price. "
        "Not a reserves or economic certification — validate with a full type-curve and price deck."
    )

with tabs[11]:
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
                    build_context(a, anomalies, econ, mb_result, st.session_state.get("aof_result"), oil_pvt_summary, fractional_flow_summary),
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
**Important:** REX is a prototype decision-support tool. Its anomaly flags, gas PVT correlations
(Dranchuk & Abou-Kassem Z-factor, Standing pseudo-criticals, Lee-Gonzalez-Eakin viscosity), oil
PVT correlations (Standing Rs/Bo, Beggs-Robinson viscosity, Vasquez-Beggs undersaturated
compressibility — the black-oil correlation set compiled in McCain's *Properties of Petroleum
Fluids*), Buckley-Leverett fractional flow / Welge shock-front analysis, P/Z material balance /
OGIP estimate, Rawlins-Schellhardt deliverability (AOF), decline-curve forecasts, and economic
calculations are screening-level outputs and should be validated by a qualified engineer using
field-specific data, lab-measured PVT (if available), and standard industry workflows.
""")
