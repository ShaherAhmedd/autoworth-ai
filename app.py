from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from src.autoworth import load_dataset_files, load_local_dataset, load_zip, train_model, make_prediction_row


st.set_page_config(
    page_title="AutoWorth | Used Car Price & Deal Advisor",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ink:#17231e; --muted:#66736c; --paper:#f5f6f2; --green:#147d5a; }
.stApp { background:var(--paper); color:var(--ink); font-family:'DM Sans',sans-serif; }
h1,h2,h3 { font-family:'Manrope',sans-serif; letter-spacing:-.035em; }
.block-container { padding-top:2.2rem; padding-bottom:3rem; max-width:1320px; }
.hero { padding:2rem 2.2rem; border-radius:22px; background:linear-gradient(115deg,#14291f,#1c5941); color:#fff; margin-bottom:1.5rem; }
.hero .eyebrow { color:#b3dfc8; font-size:.78rem; font-weight:700; letter-spacing:.13em; text-transform:uppercase; }
.hero h1 { color:#fff; font-size:2.65rem; margin:.35rem 0 .5rem; }
.hero p { color:#d5e7dd; max-width:760px; font-size:1.05rem; margin:0; }
.small-note { color:#78847d; font-size:.85rem; }
div[data-testid="stMetric"] { border:1px solid #e0e6df; background:#fff; padding:1rem 1.1rem; border-radius:15px; }
div[data-testid="stMetricLabel"] p { color:#66736c; }
div[data-testid="stButton"] button[kind="primary"] { background:#147d5a; border-color:#147d5a; }
[data-testid="stSidebar"] { background:#edf1ea; }
</style>
<div class="hero">
  <div class="eyebrow">Used car intelligence · UK listings</div>
  <h1>Know what a car is worth.</h1>
  <p>Estimate a used car’s market price, compare it with the asking price, and explore what shapes value across nine UK brands.</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def cached_training(_frame: pd.DataFrame, fingerprint: str):
    return train_model(_frame)


def get_frame() -> pd.DataFrame | None:
    if "dataset" in st.session_state:
        return st.session_state.dataset
    local_error = None
    try:
        local = load_local_dataset(Path(__file__).parent / "data")
    except ValueError as error:
        local = None
        local_error = str(error)
    if local is not None:
        st.session_state.dataset = local
        st.session_state.dataset_source = "Repository data folder"
        return local

    with st.sidebar:
        st.markdown("### Add the project data")
        st.caption("Use the nine manufacturer CSV files from the notebook. Add them to `data/` before deployment, or upload one ZIP here.")
        if local_error:
            st.warning(local_error)
        archive = st.file_uploader("Upload a ZIP of the nine CSV files", type=["zip"])
        if archive is not None:
            try:
                frame = load_zip(archive.getvalue())
                st.session_state.dataset = frame
                st.session_state.dataset_source = "Uploaded ZIP"
                st.rerun()
            except Exception as error:
                st.error(str(error))
        csv_files = st.file_uploader("Or select all nine CSV files", type=["csv"], accept_multiple_files=True)
        if csv_files:
            try:
                frame = load_dataset_files({file.name: file.getvalue() for file in csv_files})
                st.session_state.dataset = frame
                st.session_state.dataset_source = "Uploaded CSV files"
                st.rerun()
            except Exception as error:
                st.error(str(error))
    return None


with st.sidebar:
    st.markdown("## AutoWorth AI")
    st.caption("Machine learning · used car price & deal advisor")
    st.divider()

frame = get_frame()
if frame is None:
    st.info("Add the nine project CSV files to the `data` folder, or upload them in the sidebar to get started.")
    st.markdown("#### Expected files")
    st.code("audi.csv  bmw.csv  ford.csv  hyundi.csv  merc.csv\nskoda.csv  toyota.csv  vauxhall.csv  vw.csv", language="text")
    st.markdown("The original notebook’s separate `focus.csv`, `cclass.csv`, and unclean files are intentionally not used.")
    st.stop()

fingerprint = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()
if "trained_fingerprint" not in st.session_state or st.session_state.trained_fingerprint != fingerprint:
    with st.spinner("Training and evaluating the price model on the project data…"):
        try:
            model, metrics = cached_training(frame, fingerprint)
            st.session_state.model = model
            st.session_state.metrics = metrics
            st.session_state.trained_fingerprint = fingerprint
        except Exception as error:
            st.error(f"The model could not train on this dataset: {error}")
            st.stop()
model = st.session_state.model
metrics = st.session_state.metrics

tab_estimate, tab_insights, tab_project = st.tabs(["Price & deal advisor", "Market insights", "About the model"])

with tab_estimate:
    st.subheader("Describe the car")
    st.caption(f"Trained from {len(frame):,} cleaned listings · Data source: {st.session_state.get('dataset_source', 'project data')}")
    makes = sorted(frame["Make"].dropna().unique().tolist())
    with st.form("estimate_form"):
        left, right = st.columns(2, gap="large")
        with left:
            make = st.selectbox("Make", makes)
            make_rows = frame[frame["Make"] == make]
            models = sorted(make_rows["model"].dropna().unique().tolist())
            car_model = st.selectbox("Model", models)
            year_min = int(frame["year"].min())
            year_max = int(frame["year"].max())
            year = st.number_input("Year", min_value=year_min, max_value=year_max, value=min(2018, year_max), step=1)
            mileage_max = max(300_000, int(frame["mileage"].max()))
            mileage = st.number_input("Mileage", min_value=0, max_value=mileage_max, value=25_000, step=1_000, help="Enter the odometer reading in miles.")
            transmission = st.selectbox("Transmission", sorted(frame["transmission"].dropna().unique().tolist()))
        with right:
            fuel_type = st.selectbox("Fuel type", sorted(frame["fuelType"].dropna().unique().tolist()))
            tax_max = max(1_000, int(frame["tax"].max()))
            tax = st.number_input("Annual tax (£)", min_value=0, max_value=tax_max, value=min(145, tax_max), step=5)
            mpg_max = max(150, int(frame["mpg"].max()))
            mpg = st.number_input("Fuel economy (MPG)", min_value=0.0, max_value=float(mpg_max), value=55.0, step=1.0)
            engine_max = max(10.0, float(frame["engineSize"].max()))
            engine_size = st.number_input("Engine size (L)", min_value=0.0, max_value=engine_max, value=2.0, step=0.1)
            asking_price = st.number_input("Seller asking price (£)", min_value=0, max_value=250_000, value=17_000, step=250)
        submitted = st.form_submit_button("Estimate value", type="primary", use_container_width=True)

    if submitted:
        row = make_prediction_row(
            make=make, model=car_model, year=int(year), mileage=int(mileage),
            transmission=transmission, fuel_type=fuel_type, tax=float(tax),
            mpg=float(mpg), engine_size=float(engine_size),
        )
        estimate = max(0.0, float(model.predict(row)[0]))
        delta = asking_price - estimate
        percent = delta / estimate * 100 if estimate else 0.0
        col1, col2, col3 = st.columns(3)
        col1.metric("Estimated market price", f"£{estimate:,.0f}")
        col2.metric("Seller asking price", f"£{asking_price:,.0f}", f"{percent:+.1f}% vs estimate")
        col3.metric("Gap to estimate", f"£{abs(delta):,.0f}", "Below estimate" if delta < 0 else "Above estimate")
        if percent < -10:
            st.success("Great deal signal — the asking price is more than 10% below the model estimate.")
        elif percent < -5:
            st.success("Good deal signal — the asking price is 5–10% below the model estimate.")
        elif percent <= 5:
            st.info("Fair price signal — the asking price is within 5% of the model estimate.")
        elif percent <= 10:
            st.warning("Slightly overpriced — the asking price is 5–10% above the model estimate.")
        else:
            st.error("Overpriced signal — the asking price is more than 10% above the model estimate.")
        st.caption("This is a data-driven estimate, not a formal valuation. Check vehicle condition, service history, location, and current listings before buying.")

with tab_insights:
    st.subheader("What the listings show")
    summary = st.columns(3)
    summary[0].metric("Clean listings", f"{len(frame):,}")
    summary[1].metric("Brands", f"{frame['Make'].nunique()}")
    summary[2].metric("Median listing price", f"£{frame['price'].median():,.0f}")
    charts = st.columns(2, gap="large")
    with charts[0]:
        st.markdown("#### Median price by year")
        yearly = frame.groupby("year", as_index=False)["price"].median().sort_values("year")
        yearly = yearly.set_index("year")
        st.line_chart(yearly, y="price", color="#147d5a")
    with charts[1]:
        st.markdown("#### Median price by make")
        by_make = frame.groupby("Make")["price"].median().sort_values(ascending=True)
        st.bar_chart(by_make, color="#147d5a", horizontal=True)
    st.markdown("#### Mileage and price")
    sample = frame.sample(min(3_000, len(frame)), random_state=42)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(sample["mileage"], sample["price"], alpha=.28, s=12, color="#147d5a")
    ax.set_xlabel("Mileage (miles)")
    ax.set_ylabel("Listing price (£)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.15)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with tab_project:
    st.subheader("A complete regression workflow")
    st.markdown("The app follows the project notebook: it combines nine manufacturer datasets, removes exact duplicates and records after 2021, engineers age and mileage-per-year features, then trains a Random Forest regression pipeline.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Test R²", f"{metrics['r2']:.3f}")
    m2.metric("Test MAE", f"£{metrics['mae']:,.0f}")
    m3.metric("Test RMSE", f"£{metrics['rmse']:,.0f}")
    st.caption(f"Metrics use a held-out test split of {int(metrics['test_rows']):,} listings. MAE and RMSE summarize errors on historical listings; they do not guarantee an individual car's sale price.")
    st.markdown("#### What goes into a prediction")
    st.write("Make, model, year, mileage, transmission, fuel type, tax, MPG, and engine size. Seller asking price is used only after the estimated value is calculated to create the deal signal.")
    st.markdown("#### Data cleaning choices")
    st.write("Exact duplicates are removed and the impossible 2060 year is excluded. Unusual values such as zero engine size are kept because electric and hybrid vehicles can have them.")
    st.markdown("#### Dataset")
    st.markdown("The notebook uses the [100,000 UK Used Car Data set on Kaggle](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes). Review the dataset page’s terms before redistributing raw CSV files.")
    st.caption("For educational exploration only. Listings are historical and the estimate is not financial or purchasing advice.")
