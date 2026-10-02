from pathlib import Path
import pandas as pd
import streamlit as st

from src.carsheet import scrape_range
from src.market_data import build_training_table
from src.model import train, load

st.set_page_config(page_title="AutoValue", page_icon="🚗", layout="wide")

ROOT = Path(".")
DATA = ROOT / "data"; ART = ROOT / "artifacts"
DATA.mkdir(exist_ok=True); ART.mkdir(exist_ok=True)

st.markdown("""
<style>
.hero{padding:1.5rem 0 .7rem}.hero h1{font-size:3rem;margin:0}
.hero p{font-size:1.15rem;color:#64748b}.card{padding:1.5rem;border:1px solid #e5e7eb;
border-radius:16px;background:#fff}.price{font-size:2.7rem;font-weight:800}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>🚗 AutoValue</h1><p>Used-car valuation using Carsheet specifications + real-market data.</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("1 · Carsheet data")
    start = st.number_input("Start year", 1980, 2026, 2010)
    end = st.number_input("End year", 1980, 2026, 2025)
    delay = st.slider("Request delay", .5, 5.0, 1.5, .5)

    if st.button("Scrape Carsheet", use_container_width=True):
        if start > end:
            st.error("Start year must be <= end year.")
        else:
            bar = st.progress(0); msg = st.empty()
            def progress(p, y):
                bar.progress(p); msg.write(f"Collecting {y}…")
            try:
                with st.spinner("Collecting Carsheet specifications…"):
                    df, errors = scrape_range(int(start), int(end), delay, progress)
                df.to_csv(DATA/"carsheet.csv", index=False)
                st.session_state["carsheet"] = df
                st.success(f"{len(df):,} Carsheet rows collected.")
                if errors: st.warning(f"{len(errors)} year(s) failed.")
            except Exception as e: st.error(str(e))

    if "carsheet" not in st.session_state and (DATA/"carsheet.csv").exists():
        st.session_state["carsheet"] = pd.read_csv(DATA/"carsheet.csv")

    st.header("2 · Used-market data")
    st.caption("Upload actual listings/transactions to teach the model what cars sell for today.")
    market_file = st.file_uploader("Used-market CSV", type="csv")
    if market_file:
        try:
            market = pd.read_csv(market_file)
            st.session_state["market"] = market
            st.success(f"{len(market):,} market rows loaded.")
        except Exception as e: st.error(str(e))

    st.header("3 · Train")
    if st.button("Train valuation model", type="primary", use_container_width=True):
        if "carsheet" not in st.session_state:
            st.error("Scrape or load Carsheet data first.")
        elif "market" not in st.session_state:
            st.error("Upload used-market data first. Carsheet alone is mainly specifications/MSRP and is not enough for a current used-price model.")
        else:
            try:
                with st.spinner("Merging datasets and training…"):
                    training = build_training_table(st.session_state["carsheet"], st.session_state["market"])
                    if len(training) < 20:
                        raise ValueError(f"Only {len(training)} usable matched market rows were found. At least 20 are needed.")
                    training.to_csv(DATA/"training_data.csv", index=False)
                    meta = train(training, ART)
                    st.session_state["meta"] = meta
                st.success(f"Trained on {meta['training_rows']:,} market observations.")
            except Exception as e:
                st.error(str(e))

if "meta" not in st.session_state and (ART/"metadata.json").exists():
    try:
        _, st.session_state["meta"] = load(ART)
    except Exception:
        pass

if "carsheet" in st.session_state:
    c1,c2,c3 = st.columns(3)
    df=st.session_state["carsheet"]
    c1.metric("Carsheet vehicles", f"{len(df):,}")
    c2.metric("Makes", df["Make"].nunique() if "Make" in df else 0)
    c3.metric("Years", f"{int(df.Year.min())}–{int(df.Year.max())}" if "Year" in df and df.Year.notna().any() else "—")

if "meta" not in st.session_state:
    st.info("Scrape Carsheet, upload a used-market CSV, then train the valuation model.")
    st.markdown("""
### Why two datasets?
**Carsheet** supplies structured vehicle specifications.  
**Used-market data** supplies the variables needed for current valuation:

- Mileage / odometer
- Condition
- Location
- Accident history
- Listing date
- Actual asking/selling price

This separation prevents the model from treating MSRP as if it were a current used-car transaction price.
""")
    st.stop()

model, meta = load(ART)

st.divider()
st.subheader("Estimate a used-car price")

with st.form("predict"):
    vals={}
    cols=st.columns(2)
    for i,f in enumerate(meta["features"]):
        with cols[i%2]:
            if f in meta["numeric_features"]:
                vals[f]=st.number_input(f, value=0.0, key="n_"+f)
            else:
                vals[f]=st.text_input(f, key="t_"+f)
    submit=st.form_submit_button("Estimate price", type="primary", use_container_width=True)

if submit:
    row=pd.DataFrame([vals], columns=meta["features"])
    try:
        p=float(model.predict(row)[0])
        st.markdown(f'<div class="card"><div>Estimated used-car price</div><div class="price">${p:,.0f}</div></div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.divider()
st.subheader("Model evaluation")
a,b,c=st.columns(3)
a.metric("Model",meta["best_model"])
b.metric("MAE",f"${meta['metrics']['MAE']:,.0f}")
c.metric("R²",f"{meta['metrics']['R2']:.3f}")
st.caption(f"Training observations: {meta['training_rows']:,}. Metrics are from a held-out test set.")
