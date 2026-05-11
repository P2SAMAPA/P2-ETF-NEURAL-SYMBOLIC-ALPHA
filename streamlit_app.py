import streamlit as st
import pandas as pd
import json
import plotly.express as px
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="Neural Symbolic Alpha", layout="wide")
st.title("🧠 Neural Symbolic Alpha – Discover Trading Formulas")
st.caption("PySR symbolic regression | Closed‑form alpha factors | Predicts next‑day returns")

OUTPUT_REPO = config.OUTPUT_REPO
HF_TOKEN = config.HF_TOKEN

@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        files = [f['name'] for f in fs.ls(f"datasets/{OUTPUT_REPO}", detail=True, recursive=True) if f['type'] == 'file']
        return files
    except Exception as e:
        return [f"Error: {e}"]

def find_latest_json(files):
    json_files = [f for f in files if f.endswith('.json') and 'symbolic_alpha' in f]
    if not json_files:
        return None
    json_files.sort(reverse=True)
    return json_files[0]

@st.cache_data(ttl=3600)
def load_json(path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

files = list_repo_files()
latest = find_latest_json(files)
if not latest:
    st.error("No results found. Run trainer first.")
    st.stop()

data = load_json(latest)
if "error" in data:
    st.error(f"Error loading JSON: {data['error']}")
    st.stop()

st.sidebar.header("ℹ️ Info")
st.sidebar.write(f"**Run date:** {data['run_date']}")
st.sidebar.write(f"**Next trading day:** {next_trading_day()}")
st.sidebar.write("**Method:** PySR symbolic regression (GP + neural priors)")

universes = data["universes"]
if not universes:
    st.warning("No universe data.")
    st.stop()

st.header("📈 Top Discovered Alpha Factors (by combined return-predictability score)")

for universe_name, uni_data in universes.items():
    top = uni_data.get("top_expressions", [])
    if not top:
        continue
    st.subheader(f"🌍 {universe_name}")
    cols = st.columns(min(len(top), 3))
    for i, item in enumerate(top):
        with cols[i]:
            ticker = item["ticker"]
            # Support both new 'combined_score' and old 'correlation'
            score = item.get("combined_score", item.get("correlation", 0.0))
            st.metric(ticker, f"score = {score:.4f}", "alpha strength")
            # Show the expression if available
            ticker_data = uni_data["all_tickers"].get(ticker, {})
            expr = ticker_data.get("expression", "N/A")
            complexity = ticker_data.get("complexity", "?")
            with st.expander(f"Formula for {ticker}"):
                st.code(expr, language="python")
                st.caption(f"Complexity: {complexity}")
    st.divider()

# Detailed view
universe_names = list(universes.keys())
selected = st.selectbox("Select Universe for detailed table", universe_names)
if selected:
    uni_data = universes[selected]
    all_tickers = uni_data.get("all_tickers", {})
    if all_tickers:
        rows = []
        for ticker, info in all_tickers.items():
            rows.append({
                "ETF": ticker,
                "Validation Corr": info.get("validation_correlation", 0.0),
                "Avg Return": info.get("avg_return", 0.0),
                "Combined Score": info.get("combined_score", 0.0),
                "Complexity": info.get("complexity", 0),
                "MSE": info.get("mse", 0.0),
                "Window (days)": info.get("selected_window", 0),
                "Formula snippet": info.get("expression", "")[:50] + "..."
            })
        df = pd.DataFrame(rows).sort_values("Combined Score", ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No data for this universe.")

st.caption("Best expression per ETF is selected by validation correlation. The combined score (correlation × average return) ranks ETFs for higher return potential.")
