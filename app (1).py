import io
import math
import re
from datetime import date, datetime
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import sympy as sp
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Hispanic Heritage Historical Data Explorer", layout="wide")
st.title("Hispanic Heritage Historical Data Explorer")
st.caption("HISPANIC HERITAGE DATA LAB • Historical data, modeling, and calculus")

COUNTRIES = {
    "Mexico": "MX", "Brazil": "BR", "Argentina": "AR", "Colombia": "CO",
    "Chile": "CL", "Peru": "PE", "Venezuela": "VE", "Ecuador": "EC", "Uruguay": "UY"
}
INDICATORS = {
    "Life expectancy at birth": ("SP.DYN.LE00.IN", "years", "Expected years a newborn would live if current age-specific mortality rates remained constant."),
    "Unemployment, national estimate": ("SL.UEM.TOTL.NE.ZS", "% of labor force", "Share of the labor force without work but available for and seeking employment, using national estimates."),
    "GDP per capita": ("NY.GDP.PCAP.CD", "current US$ per person", "Gross domestic product divided by midyear population; this is not average wealth or household income."),
    "Crude birth rate": ("SP.DYN.CBRT.IN", "births per 1,000 people per year", "Annual live births per 1,000 people; this is a rate, not a birth count."),
    "Net migration": ("SM.POP.NETM", "people", "Net number of migrants over the period; it is not an emigration count."),
    "Intentional homicide rate": ("VC.IHR.PSRC.P5", "per 100,000 people", "Intentional homicides per 100,000 people."),
    "Consumer price inflation": ("FP.CPI.TOTL.ZG", "% annual change", "Annual percentage change in the consumer price index."),
    "Central government debt": ("GC.DOD.TOTL.GD.ZS", "% of GDP", "Central government debt as a share of GDP."),
    "Total reserves including gold": ("FI.RES.TOTL.CD", "current US$", "Total reserves including gold, valued in current US dollars."),
    "External balance on goods/services": ("NE.RSB.GNFS.ZS", "% of GDP", "External balance on goods and services as a share of GDP.")
}
CONTEXTS = {
    "Business": ("Revenue", "US dollars", "Time", "years", "A revenue or sales trend; treat predictions as a mathematical scenario, not a business forecast."),
    "Engineering": ("Displacement", "meters", "Time", "seconds", "Position/displacement over time; first derivative is velocity and second derivative is acceleration."),
    "Medical": ("Concentration", "mg/L", "Time", "hours", "A concentration curve is illustrative only and is not medical advice."),
    "Sports science": ("Distance", "meters", "Time", "seconds", "Distance or performance over time; interpret rates in the chosen units."),
    "Biology": ("Population", "organisms", "Time", "days", "Population trends can be modeled, but real populations may not follow a polynomial."),
    "Bioengineering": ("Signal", "arbitrary units", "Time", "seconds", "A simplified signal model; units depend on the experiment."),
    "Economic forecasting": ("Indicator", "indicator units", "Year", "years", "A mathematical trend is not a validated economic forecast."),
    "Weather": ("Temperature", "°C", "Time", "hours", "A local temperature trend is illustrative and not a weather forecast."),
    "Custom": ("Output", "output units", "Input", "input units", "Describe your own relationship and units.")
}
SUGGESTED = {
    "Business": "a*x**2 + b*x + c", "Engineering": "a*x**2 + b*x + c",
    "Medical": "a*exp(-k*x) + c", "Sports science": "a*sin(b*x) + c",
    "Biology": "a*exp(k*x) + c", "Bioengineering": "a*sin(b*x) + c",
    "Economic forecasting": "a*x**2 + b*x + c", "Weather": "a*sin(b*x) + c", "Custom": "a*x + b"
}
WB_URL = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}?date={start}:{end}&format=json&per_page=200"

def fetch_world_bank(country, indicator, start, end):
    """Retrieve authentic World Bank observations; missing values remain missing."""
    url = WB_URL.format(country=country, indicator=indicator, start=start, end=end)
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
            return pd.DataFrame(columns=["year", "value"]), url, "The API returned an empty or unexpected response."
        rows = []
        for item in payload[1]:
            if item.get("date") and item.get("value") is not None:
                try:
                    rows.append({"year": int(item["date"]), "value": float(item["value"])})
                except (ValueError, TypeError):
                    continue
        df = pd.DataFrame(rows, columns=["year", "value"]).drop_duplicates("year").sort_values("year")
        msg = None if not df.empty else "The API returned no non-null observations for this selection."
        return df.reset_index(drop=True), url, msg
    except requests.RequestException as exc:
        return pd.DataFrame(columns=["year", "value"]), url, f"World Bank request failed: {exc}"
    except (ValueError, KeyError, TypeError) as exc:
        return pd.DataFrame(columns=["year", "value"]), url, f"Could not parse API response: {exc}"

def valid_data(df, indicator_name):
    errors = []
    if df.empty:
        return ["There are no rows to fit."]
    if df["year"].isna().any() or not np.all(np.equal(df["year"].dropna(), np.floor(df["year"].dropna()))):
        errors.append("Years must be integers.")
    if df["year"].duplicated().any():
        errors.append("Years must be unique.")
    if df["value"].isna().any() or not np.isfinite(pd.to_numeric(df["value"], errors="coerce")).all():
        errors.append("Values must be numeric and finite.")
    if indicator_name == "Life expectancy at birth" and ((df["value"] < 0) | (df["value"] > 125)).any():
        errors.append("Life expectancy values must be between 0 and 125 years.")
    if indicator_name in ("Unemployment, national estimate", "Central government debt") and (df["value"] < 0).any():
        errors.append("Values for this indicator cannot be negative.")
    return errors

def fit_polynomial(df, degree):
    d = df.sort_values("year").copy()
    base = int(d["year"].min())
    t = d["year"].to_numpy(dtype=float) - base
    y = d["value"].to_numpy(dtype=float)
    if len(np.unique(d["year"])) < degree + 1:
        raise ValueError(f"Degree {degree} requires at least {degree + 1} distinct years; only {len(np.unique(d['year']))} available.")
    coeff = np.polyfit(t, y, degree)
    model = np.poly1d(coeff)
    pred = model(t)
    return base, coeff, model, pred

def polynomial_text(coeff, base, variable="t"):
    terms = []
    degree = len(coeff) - 1
    for i, c in enumerate(coeff):
        power = degree - i
        if abs(c) < 1e-12:
            continue
        if power == 0:
            term = f"{c:.8g}"
        elif power == 1:
            term = f"{c:.8g}{variable}"
        else:
            term = f"{c:.8g}{variable}^{power}"
        terms.append(term)
    return " + ".join(terms).replace("+ -", "- ") if terms else "0"

def safe_expression(expr, x):
    allowed = {"x": x, "a": sp.Symbol("a"), "b": sp.Symbol("b"), "c": sp.Symbol("c"),
               "k": sp.Symbol("k"), "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
               "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt, "pi": sp.pi}
    try:
        parsed = sp.sympify(expr, locals=allowed, evaluate=True)
        symbols = parsed.free_symbols
        allowed_symbols = {x, sp.Symbol("a"), sp.Symbol("b"), sp.Symbol("c"), sp.Symbol("k")}
        if not symbols.issubset(allowed_symbols):
            raise ValueError("Use only x and parameters a, b, c, k.")
        return parsed
    except Exception as exc:
        raise ValueError(f"Could not read the function. Use notation like a*x**2+b*x+c. Details: {exc}")

def fmt(v, digits=6):
    try:
        return f"{float(v):,.{digits}g}"
    except Exception:
        return str(v)

def export_excel(summary, data, calculations):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        pd.DataFrame([summary]).to_excel(writer, sheet_name="Summary", index=False)
        data.to_excel(writer, sheet_name="Working data", index=False)
        calculations.to_excel(writer, sheet_name="Calculated years", index=False)
    return buf.getvalue()

def export_pdf(summary, data, calculations, fig):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(letter), rightMargin=28, leftMargin=28, topMargin=28, bottomMargin=28)
    styles = getSampleStyleSheet()
    story = [Paragraph("Hispanic Heritage Historical Data Explorer", styles["Title"]), Spacer(1, 8)]
    for k, v in summary.items():
        story.append(Paragraph(f"<b>{k}:</b> {str(v)[:500]}", styles["BodyText"]))
    story.append(Spacer(1, 8))
    img = io.BytesIO()
    fig.savefig(img, format="png", dpi=140, bbox_inches="tight")
    img.seek(0)
    story.append(RLImage(img, width=650, height=340))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Working data preview", styles["Heading2"]))
    preview = data.head(15).round(5)
    table_data = [list(preview.columns)] + preview.astype(str).values.tolist()
    t = Table(table_data, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("GRID",(0,0),(-1,-1),0.3,colors.grey),("FONTSIZE",(0,0),(-1,-1),7)]))
    story.append(t)
    story.append(Spacer(1, 8))
    story.append(Paragraph("Calculated years", styles["Heading2"]))
    calc_preview = calculations.head(30)
    if not calc_preview.empty:
        ct = Table([list(calc_preview.columns)] + calc_preview.astype(str).values.tolist(), repeatRows=1)
        ct.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("GRID",(0,0),(-1,-1),0.3,colors.grey),("FONTSIZE",(0,0),(-1,-1),7)]))
        story.append(ct)
    doc.build(story)
    return buf.getvalue()

# Controls
with st.sidebar:
    st.header("Data and model")
    country_name = st.selectbox("Country", list(COUNTRIES))
    indicator_name = st.selectbox("Indicator", list(INDICATORS))
    degree = st.slider("Polynomial degree", 3, 8, 3)
    sample_increment = st.slider("Graph sample increment (years)", 1, 10, 2)
    horizon = st.slider("Forward display horizon (years)", 0, 50, 10)
    compare_country = st.selectbox("Comparison country", ["None"] + [c for c in COUNTRIES if c != country_name])
    printer_mode = st.checkbox("Printer-friendly view", False)
    st.caption("Brazil is included as a Latin American comparison; it is not a Spanish-speaking country.")

current_year = date.today().year
start_year, end_year = current_year - 70, current_year - 1
country_code = COUNTRIES[country_name]
indicator_code, units, definition = INDICATORS[indicator_name]
cache_key = f"{country_code}_{indicator_code}_{start_year}_{end_year}"
if st.session_state.get("data_key") != cache_key:
    source_df, api_url, api_message = fetch_world_bank(country_code, indicator_code, start_year, end_year)
    st.session_state["source_df"] = source_df.copy()
    st.session_state["data_key"] = cache_key
    st.session_state["api_url"] = api_url
    st.session_state["api_message"] = api_message
    st.session_state.pop("edited_df", None)
    st.session_state.pop("calc_single", None)
    st.session_state.pop("calc_batch", None)
source_df = st.session_state["source_df"].copy()
if "edited_df" not in st.session_state:
    st.session_state["edited_df"] = source_df.copy()
edited_df = st.session_state["edited_df"].copy()
edited_status = not edited_df.equals(source_df)
st.write(f"**Research question starter:** How has {indicator_name.lower()} changed in {country_name}, and how does its historical pattern compare with another Latin American country?")
st.write(f"**Source:** World Bank • Indicator `{indicator_code}` • Units: {units}")
st.write(f"**Definition:** {definition}")
st.write(f"**Requested period:** {start_year}–{end_year} • **Retrieved:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
st.markdown(f"[Open World Bank API source]({st.session_state['api_url']})")
if st.session_state.get("api_message"):
    st.warning(st.session_state["api_message"])
if edited_status:
    st.warning("User-modified scenario: the working table differs from the original source data.")
else:
    st.success("Working data currently matches the retrieved source data.")

tab_data, tab_model, tab_functions, tab_exports = st.tabs(["Data", "Historical model", "Function lab", "Exports & deployment"])
with tab_data:
    st.subheader("Coverage and editable working data")
    if not source_df.empty:
        st.write(f"Actual non-null coverage: **{source_df.year.min()}–{source_df.year.max()}**, **{len(source_df)} observations** (not guaranteed to be 70).")
    else:
        st.info("No source observations are available for this selection. Check the API link or try another indicator/country.")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Reset to source data"):
            st.session_state["edited_df"] = source_df.copy()
            st.session_state.pop("calc_single", None)
            st.session_state.pop("calc_batch", None)
            st.rerun()
    with col_b:
        st.caption("Edit year/value rows below. Years must be unique integers; values must be finite numbers.")
    edited_input = st.data_editor(edited_df, num_rows="dynamic", use_container_width=True, key=f"editor_{cache_key}")
    if st.button("Apply working table edits"):
        try:
            temp = edited_input.copy()
            temp["year"] = pd.to_numeric(temp["year"], errors="raise")
            temp["value"] = pd.to_numeric(temp["value"], errors="raise")
            errs = valid_data(temp, indicator_name)
            if errs:
                st.error(" • ".join(errs))
            else:
                temp["year"] = temp["year"].astype(int)
                temp = temp.sort_values("year").reset_index(drop=True)
                st.session_state["edited_df"] = temp
                st.session_state.pop("calc_single", None)
                st.session_state.pop("calc_batch", None)
                st.success("Working data updated.")
                st.rerun()
        except Exception as exc:
            st.error(f"Could not apply edits: {exc}")
    st.dataframe(st.session_state["edited_df"], use_container_width=True)
    if compare_country != "None":
        other_df, other_url, other_msg = fetch_world_bank(COUNTRIES[compare_country], indicator_code, start_year, end_year)
        st.subheader(f"Country comparison: {country_name} and {compare_country}")
        figc, axc = plt.subplots(figsize=(9, 3.5))
        if not source_df.empty:
            axc.plot(source_df.year, source_df.value, "o-", label=country_name)
        if not other_df.empty:
            axc.plot(other_df.year, other_df.value, "o-", label=compare_country)
        axc.set_xlabel("Year"); axc.set_ylabel(units); axc.legend(); axc.grid(True, alpha=.25)
        st.pyplot(figc)
        st.caption(f"Different data coverage and missing years can affect comparisons. Sources: [primary]({st.session_state['api_url']}) · [comparison]({other_url})")
        if other_msg:
            st.info(f"{compare_country}: {other_msg}")

with tab_model:
    st.subheader("Centered-time polynomial fit")
    data = st.session_state["edited_df"].copy()
    errors = valid_data(data, indicator_name)
    fit_ok = False
    if errors:
        st.error("Data validation: " + " ".join(errors))
    elif len(data) < degree + 1:
        st.error(f"Degree {degree} requires at least {degree + 1} distinct observations.")
    else:
        try:
            base, coeff, model, fitted = fit_polynomial(data, degree)
            fit_ok = True
            data = data.sort_values("year").reset_index(drop=True)
            years = data.year.to_numpy()
            values = data.value.to_numpy()
            residuals = values - fitted
            r2 = r2_score(values, fitted) if len(values) > 1 else float("nan")
            mae = mean_absolute_error(values, fitted)
            tvar = sp.Symbol("t")
            eq = polynomial_text(coeff, base)
            summary = {
                "Country": country_name, "Country code": country_code, "Indicator": indicator_name,
                "Indicator code": indicator_code, "Units": units, "Definition": definition,
                "Requested years": f"{start_year}-{end_year}", "Actual coverage": f"{years.min()}-{years.max()}",
                "Observation count": len(data), "Base year": base, "Degree": degree,
                "Equation": f"y = {eq}, where t = year - {base}",
                "In-sample R-squared": round(float(r2), 6), "In-sample MAE": round(float(mae), 6),
                "Mean observed value": round(float(np.mean(values)), 6),
                "Data status": "User-modified scenario" if edited_status else "World Bank source data",
                "Retrieved": datetime.now().strftime("%Y-%m-%d %H:%M")
            }
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Observations", len(data)); m2.metric("In-sample R²", fmt(r2)); m3.metric("In-sample MAE", fmt(mae)); m4.metric("Mean", f"{fmt(np.mean(values))} {units}")
            st.markdown(f"**Model:** \(y = {eq}\), where \(t = year - {base}\)")
            st.caption("R² and MAE are in-sample fit metrics, not proof of predictive accuracy. A polynomial can oscillate and extrapolate implausibly.")
            st.write(f"**Observed range:** {years.min()}–{years.max()} • **Fitted range:** {years.min()}–{years.max()} • **Data status:** {summary['Data status']}")
            res_df = pd.DataFrame({"year": years, "observed": values, "fitted_at_observed_year": fitted, "residual (observed - fitted)": residuals})
            with st.expander("Residuals and fitted values at observed years"):
                st.dataframe(res_df, use_container_width=True)
            # Chronological holdout when at least degree+3 observations
            if len(data) >= degree + 3:
                split = max(degree + 1, int(len(data) * .8))
                train = data.iloc[:split]
                test = data.iloc[split:]
                if len(test) and len(train) >= degree + 1:
                    hb, hc, hm, _ = fit_polynomial(train, degree)
                    hp = hm(test.year.to_numpy(dtype=float) - hb)
                    st.write(f"**Chronological holdout MAE** (last {len(test)} observations; trained only on earlier years): {fmt(mean_absolute_error(test.value, hp))} {units}. This is separate from in-sample metrics.")
            x_smooth = np.linspace(years.min(), years.max() + horizon, max(200, (years.max() - years.min() + horizon) * 3 + 1))
            y_smooth = model(x_smooth - base)
            sample_years = np.arange(years.min(), years.max() + horizon + 1, sample_increment)
            sample_values = model(sample_years - base)
            fig, ax = plt.subplots(figsize=(10, 4.5))
            ax.scatter(years, values, label="Observed World Bank / working data", zorder=3)
            ax.plot(x_smooth[x_smooth <= years.max()], y_smooth[x_smooth <= years.max()], label="Fitted curve (historical)")
            if horizon > 0:
                ax.plot(x_smooth[x_smooth > years.max()], y_smooth[x_smooth > years.max()], "--", label="Polynomial extrapolation")
                ax.axvline(years.max(), color="gray", linestyle=":", label="Historical boundary")
            ax.plot(sample_years, sample_values, "x", label=f"Sample markers every {sample_increment} year(s)")
            if st.session_state.get("calc_single"):
                qy, qv = st.session_state["calc_single"]
                ax.scatter([qy], [qv], marker="*", s=130, label="Single-year estimate")
            if st.session_state.get("calc_batch") is not None and not st.session_state["calc_batch"].empty:
                bc = st.session_state["calc_batch"]
                ax.scatter(bc["year"], bc["estimate"], marker="D", s=25, label="Batch estimates")
            ax.set_xlabel("Year"); ax.set_ylabel(units); ax.legend(loc="best", fontsize=8); ax.grid(True, alpha=.25)
            st.pyplot(fig)
            st.subheader("Calculate a single year")
            qyear = st.number_input("Year (1950–2100)", min_value=1950, max_value=2100, value=int(min(max(int(years.max()),1950),2100)), step=1)
            if st.button("Calculate Value"):
                val = float(model(float(qyear) - base))
                st.session_state["calc_single"] = (int(qyear), val)
            if st.session_state.get("calc_single"):
                qy, qv = st.session_state["calc_single"]
                classification = "within observed fitted range" if years.min() <= qy <= years.max() else "outside observed fitted range (extrapolation)"
                actual_match = data.loc[data.year == qy, "value"]
                st.write(f"**Year {qy}:** {fmt(qv)} {units} • {classification}.")
                if not actual_match.empty:
                    st.write(f"Observed value in working data: **{fmt(actual_match.iloc[0])} {units}**. The estimate is a fitted value, not the observation.")
                if qy < years.min() or qy > years.max():
                    st.warning("Extrapolations are especially uncertain. Check whether the result is plausible; the app does not silently clip it.")
            st.subheader("Batch calculations")
            mode = st.radio("Batch input type", ["Year range", "Custom years"], horizontal=True)
            if mode == "Year range":
                c1, c2, c3 = st.columns(3)
                with c1: bstart = st.number_input("Start year", 1950, 2100, int(max(1950, years.min())), key="bstart")
                with c2: bend = st.number_input("End year", 1950, 2100, int(min(2100, years.max())), key="bend")
                with c3: bstep = st.number_input("Step (1–10 years)", 1, 10, 1, key="bstep")
                if st.button("Calculate batch range"):
                    if bstart > bend:
                        st.error("Start year must be less than or equal to end year.")
                    else:
                        byears = list(range(int(bstart), int(bend)+1, int(bstep)))
                        st.session_state["calc_batch"] = pd.DataFrame({"year": byears, "estimate": [float(model(y-base)) for y in byears], "units": units})
            else:
                custom = st.text_input("Comma-separated years", "2000, 2010, 2020")
                if st.button("Calculate custom years"):
                    try:
                        byears = [int(x.strip()) for x in custom.split(",") if x.strip()]
                        invalid = [x for x in byears if x < 1950 or x > 2100]
                        if not byears or invalid:
                            raise ValueError(f"Enter one or more integer years from 1950 to 2100. Invalid: {invalid}")
                        st.session_state["calc_batch"] = pd.DataFrame({"year": byears, "estimate": [float(model(y-base)) for y in byears], "units": units})
                    except Exception as exc:
                        st.error(str(exc))
            if st.session_state.get("calc_batch") is not None:
                st.dataframe(st.session_state["calc_batch"], use_container_width=True)
            # course-level analysis
            st.subheader("Course-level interpretation")
            pathway = st.selectbox("Mathematics pathway", ["Honors Algebra 2", "Precalculus", "Calculus"])
            st.write(f"- **Domain in this project:** all real-valued years can be substituted algebraically, but historical evidence only covers {years.min()}–{years.max()}.")
            st.write(f"- **Range:** the fitted model's range depends on its degree and domain; the plotted values show the selected display interval.")
            st.write("- **End behavior:** use the leading coefficient and degree to understand polynomial behavior far beyond the data; distant extrapolation is not a reliable historical claim.")
            if pathway in ("Precalculus", "Calculus"):
                deriv = np.polyder(model)
                st.write(f"- **Changing average rates:** the derivative polynomial describes how the fitted rate changes; turning points can make the trend reverse.")
                st.write(f"- **Degree/coefficient effects:** degree controls possible turning behavior; coefficient signs and magnitudes affect curvature and scale.")
            if pathway == "Calculus":
                d1 = np.polyder(model); d2 = np.polyder(model, 2)
                st.markdown(f"**First derivative:** \(f'(t) = {polynomial_text(d1.c, base='')}\) (value-units per year).")
                st.markdown(f"**Second derivative:** \(f''(t) = {polynomial_text(d2.c, base='')}\) (value-units per year²).")
                roots1 = np.roots(d1); real_crit = sorted([float(r.real + base) for r in roots1 if abs(r.imag) < 1e-7 and years.min() <= r.real+base <= years.max()])
                st.write(f"Numerical critical points inside observed years: {', '.join(fmt(x,5) for x in real_crit) if real_crit else 'none found'}.")
                roots2 = np.roots(d2); inflect = sorted([float(r.real + base) for r in roots2 if abs(r.imag) < 1e-7 and years.min() <= r.real+base <= years.max()])
                st.write(f"Candidate inflection points inside observed years: {', '.join(fmt(x,5) for x in inflect) if inflect else 'none found'}. Candidates require a concavity-change check.")
            st.info("Uncertainty is described using residuals and MAE only. These are not validated 95% prediction intervals. Historical associations do not establish causes.")
            st.session_state["latest_summary"] = summary
            st.session_state["latest_data"] = data
            st.session_state["latest_fig"] = fig
        except Exception as exc:
            st.error(f"Could not fit this model: {exc}")
            fit_ok = False
    if not fit_ok:
        st.info("Choose a valid dataset and a degree supported by the number of distinct observations before calculating a fit.")

with tab_functions:
    st.subheader("Function lab: choose a context and build a function")
    context = st.selectbox("Context", list(CONTEXTS))
    dep_name, dep_units, indep_name, indep_units, context_desc = CONTEXTS[context]
    c1, c2 = st.columns(2)
    with c1:
        dep_name = st.text_input("Dependent variable name", dep_name)
        dep_units = st.text_input("Dependent units", dep_units)
        indep_name = st.text_input("Independent variable name", indep_name)
        indep_units = st.text_input("Independent units", indep_units)
    with c2:
        context_desc = st.text_area("Context description", context_desc)
        function_mode = st.selectbox("Function source", ["Suggested function", "Custom function", "Curve of best fit from selected data"])
        expr_default = SUGGESTED[context]
        expr = st.text_input("Function f(x)", expr_default if function_mode == "Suggested function" else "a*x**2+b*x+c")
        a = st.number_input("Parameter a", value=1.0)
        b = st.number_input("Parameter b", value=0.5)
        c = st.number_input("Parameter c", value=0.0)
        k = st.number_input("Parameter k", value=0.1)
    xmin, xmax = st.slider("Analysis interval", -20.0, 100.0, (0.0, 10.0), step=0.5)
    xpoint = st.slider("Current input x", float(xmin), float(xmax), float((xmin+xmax)/2), step=max((xmax-xmin)/100,0.01))
    st.caption("The prosthetic-leg use case can be modeled as a position-versus-time function in Engineering or Sports science; the function and units remain customizable. Visual scaling is determined by the selected interval and plotted values.")
    x_symbol = sp.Symbol("x")
    try:
        if function_mode == "Curve of best fit from selected data":
            fitdata = st.session_state["edited_df"].copy()
            if len(fitdata) < 2:
                raise ValueError("At least two observations are required to create a fitted curve.")
            fbase, fcoeff, fmodel, _ = fit_polynomial(fitdata, min(3, max(1, len(fitdata)-1)))
            # Transform data-year x to centered time so expression is clear.
            expr_sym = sum(sp.Float(float(cc))*((x_symbol-fbase)**(len(fcoeff)-1-i)) for i, cc in enumerate(fcoeff))
            st.write(f"Using selected data fit: t = year - {fbase}. This is a polynomial fit to the current working data.")
        else:
            expr_sym = safe_expression(expr, x_symbol).subs({sp.Symbol("a"): a, sp.Symbol("b"): b, sp.Symbol("c"): c, sp.Symbol("k"): k})
        f = sp.lambdify(x_symbol, expr_sym, modules=["numpy"])
        f1sym = sp.diff(expr_sym, x_symbol); f2sym = sp.diff(f1sym, x_symbol)
        f1 = sp.lambdify(x_symbol, f1sym, modules=["numpy"]); f2 = sp.lambdify(x_symbol, f2sym, modules=["numpy"])
        xs = np.linspace(xmin, xmax, 700)
        with np.errstate(all="ignore"):
            ys = np.asarray(f(xs), dtype=float)
            d1ys = np.asarray(f1(xs), dtype=float)
            d2ys = np.asarray(f2(xs), dtype=float)
        if ys.ndim == 0: ys = np.full_like(xs, float(ys))
        if d1ys.ndim == 0: d1ys = np.full_like(xs, float(d1ys))
        if d2ys.ndim == 0: d2ys = np.full_like(xs, float(d2ys))
        ys = np.where(np.isfinite(ys), ys, np.nan)
        d1ys = np.where(np.isfinite(d1ys), d1ys, np.nan)
        d2ys = np.where(np.isfinite(d2ys), d2ys, np.nan)
        st.markdown(f"**Function:** \(f(x) = {sp.latex(expr_sym)}\)")
        st.markdown(f"**First derivative:** \(f'(x) = {sp.latex(f1sym)}\)")
        st.markdown(f"**Second derivative:** \(f''(x) = {sp.latex(f2sym)}\)")
        show_f = st.checkbox("Show function", True)
        show_d1 = st.checkbox("Show first derivative", False)
        show_d2 = st.checkbox("Show second derivative", False)
        figf, axf = plt.subplots(figsize=(10, 4.5))
        if show_f: axf.plot(xs, ys, label=f"{dep_name} f(x)")
        if show_d1: axf.plot(xs, d1ys, label="First derivative")
        if show_d2: axf.plot(xs, d2ys, label="Second derivative")
        with np.errstate(all="ignore"):
            ypoint = float(np.asarray(f(xpoint)))
            slope = float(np.asarray(f1(xpoint)))
            curvature = float(np.asarray(f2(xpoint)))
        if np.isfinite(ypoint) and np.isfinite(slope):
            axf.scatter([xpoint], [ypoint], s=55, label="Current point")
            tangent_x = np.linspace(max(xmin, xpoint-(xmax-xmin)*.18), min(xmax, xpoint+(xmax-xmin)*.18), 50)
            axf.plot(tangent_x, ypoint+slope*(tangent_x-xpoint), "--", label="Tangent line")
        axf.set_xlabel(f"{indep_name} ({indep_units})"); axf.set_ylabel(f"{dep_name} ({dep_units})"); axf.legend(); axf.grid(True, alpha=.25)
        st.pyplot(figf)
        st.write(f"**At x = {fmt(xpoint)} {indep_units}:** f(x) = {fmt(ypoint)} {dep_units}; f′(x) = {fmt(slope)} {dep_units}/{indep_units}; f″(x) = {fmt(curvature)} {dep_units}/{indep_units}².")
        st.write(f"**In context:** {context_desc} The first derivative is the instantaneous rate of change; the second derivative describes how that rate is changing.")
        left, right = st.columns(2)
        with left:
            xa = st.slider("Secant interval start", float(xmin), float(xmax), float(xmin), step=max((xmax-xmin)/100,0.01))
        with right:
            xb = st.slider("Secant interval end", float(xmin), float(xmax), float(xmax), step=max((xmax-xmin)/100,0.01))
        if xa != xb:
            ya, yb = float(np.asarray(f(xa))), float(np.asarray(f(xb)))
            avg_rate = (yb-ya)/(xb-xa)
            st.write(f"Average rate of change on [{fmt(xa)}, {fmt(xb)}] = **{fmt(avg_rate)} {dep_units}/{indep_units}**.")
        else:
            st.warning("Choose different secant interval endpoints.")
        # Numerical roots and classification by derivative sign changes
        st.subheader("Numerical calculus analysis")
        try:
            critical = [float(sp.re(r)) for r in sp.nroots(sp.together(f1sym).as_numer_denom()[0]) if abs(float(sp.im(r))) < 1e-8 and xmin < float(sp.re(r)) < xmax]
        except Exception:
            critical = []
        crit_rows = []
        for cp in sorted(critical):
            eps = max((xmax-xmin)*1e-4, 1e-5)
            try:
                sl, sr = float(np.asarray(f1(cp-eps))), float(np.asarray(f1(cp+eps)))
                kind = "local minimum" if sl < 0 < sr else ("local maximum" if sl > 0 > sr else "stationary; no sign-change extremum")
                crit_rows.append({"x": cp, "f(x)": float(np.asarray(f(cp))), "classification": kind})
            except Exception:
                pass
        st.write("Critical-point classifications use a numerical derivative sign-change check; results are approximations.")
        st.dataframe(pd.DataFrame(crit_rows) if crit_rows else pd.DataFrame(columns=["x","f(x)","classification"]), use_container_width=True)
        try:
            inf_candidates = [float(sp.re(r)) for r in sp.nroots(sp.together(f2sym).as_numer_denom()[0]) if abs(float(sp.im(r))) < 1e-8 and xmin < float(sp.re(r)) < xmax]
        except Exception:
            inf_candidates = []
        inflection_rows = []
        for ip in sorted(inf_candidates):
            eps = max((xmax-xmin)*1e-4, 1e-5)
            try:
                cl, cr = float(np.asarray(f2(ip-eps))), float(np.asarray(f2(ip+eps)))
                if np.isfinite(cl) and np.isfinite(cr) and cl*cr < 0:
                    inflection_rows.append({"x": ip, "f(x)": float(np.asarray(f(ip))), "concavity changes": "yes"})
            except Exception:
                pass
        st.write("Inflection points are listed only when a numerical concavity sign change is detected.")
        st.dataframe(pd.DataFrame(inflection_rows) if inflection_rows else pd.DataFrame(columns=["x","f(x)","concavity changes"]), use_container_width=True)
        if function_mode == "Curve of best fit from selected data":
            st.info("The best-fit function is a polynomial approximation. Its domain as a mathematical polynomial is all real numbers, but meaningful interpretation is limited by the data and units.")
        else:
            st.info("SymPy parses the entered expression. Logarithms, rational functions, and trigonometric functions can have restricted domains or discontinuities; non-finite plot values are omitted so the graph does not intentionally connect across them.")
        # Simple synchronized manual animation controls; each step is a user action.
        st.subheader("Simulation controls")
        if "sim_x" not in st.session_state: st.session_state["sim_x"] = float(xpoint)
        speed = st.select_slider("Step size / speed", options=[0.1, 0.25, 0.5, 1.0, 2.0], value=0.5)
        ac1, ac2, ac3 = st.columns(3)
        with ac1:
            if st.button("▶ Advance one step"):
                st.session_state["sim_x"] = min(xmax, st.session_state["sim_x"] + speed)
        with ac2:
            if st.button("↺ Reset simulation"):
                st.session_state["sim_x"] = float(xmin)
        with ac3:
            st.caption("Use the input slider above to move the point; each advance synchronizes on rerun.")
        st.caption("This is step-based playback (not a background autoplay timer) so controls remain predictable in a classroom.")
    except Exception as exc:
        st.error(f"Function input error: {exc}. Try a supported expression such as a*x**2+b*x+c, a*exp(-k*x)+c, a*sin(b*x)+c, or log(x).")

with tab_exports:
    st.subheader("Download current work")
    st.write("Downloads are created from the current working data and latest successful historical model.")
    data_now = st.session_state["edited_df"].copy()
    st.download_button("Download working data CSV", data_now.to_csv(index=False).encode("utf-8"), file_name="working_data.csv", mime="text/csv")
    summary_now = st.session_state.get("latest_summary", {
        "Country": country_name, "Indicator": indicator_name, "Indicator code": indicator_code, "Units": units,
        "Data status": "User-modified scenario" if edited_status else "World Bank source data",
        "Source URL": st.session_state["api_url"], "Retrieved": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Model": "No successful model currently available; choose valid data and degree."
    })
    calcs = st.session_state.get("calc_batch", pd.DataFrame(columns=["year","estimate","units"]))
    xlsx = export_excel(summary_now, data_now, calcs)
    st.download_button("Download Excel workbook (.xlsx)", xlsx, file_name="hispanic_heritage_explorer.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    fig_export = st.session_state.get("latest_fig")
    if fig_export is None:
        fig_export, ax = plt.subplots(figsize=(8, 3))
        ax.plot(data_now.get("year", []), data_now.get("value", []), "o-")
        ax.set_xlabel("Year"); ax.set_ylabel(units); ax.set_title("Working data preview")
    pdf = export_pdf(summary_now, data_now, calcs, fig_export)
    st.download_button("Download PDF report", pdf, file_name="hispanic_heritage_report.pdf", mime="application/pdf")
    st.subheader("Deployment")
    st.markdown("""
**Local run**
1. Install Python 3.10 or newer.
2. In the project folder, run `pip install -r requirements.txt`.
3. Run `python -m streamlit run app.py`.

**GitHub + Streamlit Community Cloud**
1. Create a GitHub repository and upload `app.py`, `requirements.txt`, `README.md`, and `sources.md`.
2. Open Streamlit Community Cloud, connect GitHub, choose the repository and branch, and set the main file path to `app.py`.
3. Deploy. The World Bank API is public and requires no API key.

**Render**
- Create a **Python Web Service** connected to the repository.
- Build command: `pip install -r requirements.txt`
- Start command: `python -m streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT`
""")
    st.subheader("Verification checklist")
    st.markdown("""
- Compare an indicator code, country code, years, values, and units against the linked World Bank API.
- Test source reset after editing a year/value; the edited state should be labeled a user-modified scenario.
- Try duplicate years, non-integer years, nonnumeric values, empty data, and a degree that needs more observations than available.
- Check the displayed centered-time equation, in-sample R²/MAE, residual table, and holdout MAE label.
- Calculate one year with the single-year control and the same year in a batch; estimates should match.
- Try malformed custom function text, `log(x)` with nonpositive x values, and `1/(x-2)` around its discontinuity.
- Change country, indicator, degree, graph increment, horizon, comparison country, context, units, and visibility checkboxes; confirm outputs update.
- Open CSV in a text editor, XLSX in a spreadsheet app, and PDF in a PDF viewer.
- If deployment fails, inspect Streamlit/Render logs. Common causes: dependency install failure, wrong main-file path, network/API timeout, insufficient data for selected degree, or invalid function syntax.
""")
st.caption("Mathematical limitations: polynomial fit metrics describe the observed sample only. Extrapolation is not a validated forecast; data may be revised or missing; association does not prove causation. Custom function roots and calculus classifications are numerical approximations and can fail for difficult expressions.")
