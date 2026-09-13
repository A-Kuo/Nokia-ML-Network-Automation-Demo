"""
Streamlit frontend for NetOps-AI.

Ties the three existing pieces together into one interactive view:
- network_brain.py's CongestionPredictor (trains/loads the RandomForest)
- automation_bot.py's decision logic (threshold + config payload)
- dashboard.ts/.tsx's audit trail (read back here from the JSON payloads)
"""
import glob
import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from network_brain import CongestionPredictor
from automation_bot import NetworkAutomator

MODEL_PATH = "5g_brain.pkl"

st.set_page_config(page_title="NetOps-AI", page_icon="📡", layout="wide")
st.title("📡 NetOps-AI — Closed-Loop 5G Automation")
st.caption(
    "Sense → Reason → Act: a RandomForest congestion model driving autonomous "
    "antenna-tilt / TX-power decisions."
)


@st.cache_resource
def load_or_train_model():
    if not os.path.exists(MODEL_PATH):
        CongestionPredictor().train()
    return joblib.load(MODEL_PATH)


model = load_or_train_model()

tab_predict, tab_model, tab_log = st.tabs(
    ["🔮 Predict & Act", "📈 Model Insight", "🗂️ Automation Log"]
)

# ---------------------------------------------------------------- Predict & Act
with tab_predict:
    st.subheader("Tower Telemetry")
    col1, col2 = st.columns([1, 2])

    with col1:
        tower_id = st.text_input("Tower ID", value="T-505")
        users = st.slider("Connected Users", 50, 1000, 980)
        rsrp = st.slider("RSRP (dBm)", -130, -50, -110)
        hour = st.slider("Hour of Day", 0, 23, 18)

        run = st.button("Run Sense → Reason → Act", type="primary")

    with col2:
        if run:
            X = pd.DataFrame([{"users": users, "rsrp": rsrp, "hour": hour}])
            predicted_load = model.predict(X)[0]

            st.metric("Predicted Load", f"{predicted_load:.1%}")
            st.progress(min(float(predicted_load), 1.0))

            if predicted_load > 0.80:
                new_tilt = 6 if predicted_load > 0.9 else 3
                power_boost = 2
                payload = {
                    "target_node": f"gnb-{tower_id}",
                    "operations": [
                        {"action": "SET_REMOTE_electrical_tilt", "value": f"-{new_tilt}"},
                        {"action": "SET_transmission_power", "value": f"+{power_boost}dB"},
                    ],
                }
                st.error("⚠️ Congestion Detected — Automated Fix Triggered")
                st.json(payload)

                if st.checkbox("Write config_change file (simulate NETCONF push)"):
                    bot = NetworkAutomator.__new__(NetworkAutomator)
                    bot.model = model
                    bot.trigger_optimization(tower_id, predicted_load)
                    st.success(f"Saved config_change_{tower_id}.json")
            else:
                st.success("✅ Status: Optimal — no action needed.")
        else:
            st.info("Set tower telemetry on the left and run the loop.")

# ---------------------------------------------------------------- Model Insight
with tab_model:
    st.subheader("Feature Importance")
    importances = pd.Series(
        model.feature_importances_, index=["users", "rsrp", "hour"]
    ).sort_values()
    st.bar_chart(importances)

    st.subheader("Load Surface (users vs. RSRP, fixed hour)")
    fixed_hour = st.slider("Hour of Day (surface)", 0, 23, 18, key="surface_hour")
    users_grid = np.linspace(50, 1000, 40)
    rsrp_grid = np.linspace(-130, -50, 40)
    uu, rr = np.meshgrid(users_grid, rsrp_grid)
    grid_df = pd.DataFrame(
        {"users": uu.ravel(), "rsrp": rr.ravel(), "hour": fixed_hour}
    )
    grid_df["predicted_load"] = model.predict(grid_df)
    pivot = grid_df.pivot_table(index="rsrp", columns="users", values="predicted_load")
    st.dataframe(pivot.style.background_gradient(cmap="RdYlGn_r"), height=350)
    st.caption(
        "Each cell is the RandomForest's predicted load factor for that "
        "(users, RSRP) combination at the chosen hour."
    )

# ---------------------------------------------------------------- Automation Log
with tab_log:
    st.subheader("Config Changes Written by the Automation Bot")
    files = sorted(glob.glob("config_change_*.json"))
    if not files:
        st.info("No automated config changes yet — trigger one from the Predict & Act tab.")
    else:
        rows = []
        for f in files:
            with open(f) as fh:
                payload = json.load(fh)
            for op in payload["operations"]:
                rows.append(
                    {
                        "file": f,
                        "target_node": payload["target_node"],
                        "action": op["action"],
                        "value": op["value"],
                    }
                )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
