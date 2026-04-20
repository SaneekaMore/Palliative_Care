"""AI-Powered Palliative Care Assistant Streamlit app."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from chatbot import generate_support_response
from model import forecast_24h_trend, train_model, predict_risk
from utils import append_log, init_session_state, render_css, risk_color


st.set_page_config(
    page_title="AI Palliative Care Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_css()
init_session_state()

if "model_artifacts" not in st.session_state:
    artifacts, accuracy = train_model()
    st.session_state.model_artifacts = artifacts
    st.session_state.model_accuracy = accuracy

st.markdown(
    """
    <div class='main-header'>
        <h2 style='margin:0;'>AI Palliative Care Assistant</h2>
        <p style='margin:0.25rem 0 0 0;'>Compassionate monitoring, predictive insights, and emotional support in one place.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigate",
    ["🩺 Patient Monitoring", "📊 Dashboard", "💬 Chatbot"],
)
st.sidebar.markdown("---")
st.sidebar.caption(f"Model validation accuracy: **{st.session_state.model_accuracy:.1%}**")


if page == "🩺 Patient Monitoring":
    st.markdown("<div class='section-title'>Patient Monitoring & Risk Prediction</div><div class='divider'></div>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=180, value=88)
    with col2:
        spo2 = st.number_input("SpO2 (%)", min_value=70, max_value=100, value=95)
    with col3:
        temp = st.number_input("Temperature (°C)", min_value=34.0, max_value=42.0, step=0.1, value=37.0)
    with col4:
        pain = st.number_input("Pain Level (1-10)", min_value=1, max_value=10, value=4)

    if st.button("Analyze Patient Status", use_container_width=False):
        risk_level, prob, proba_dict = predict_risk(
            st.session_state.model_artifacts,
            int(heart_rate),
            int(spo2),
            float(temp),
            int(pain),
        )

        log_entry = append_log(int(heart_rate), int(spo2), float(temp), int(pain), risk_level, prob)
        st.session_state.logs.append(log_entry)
        st.session_state.risk_history.append({"timestamp": log_entry["timestamp"], "risk": risk_level})

        color_class = {
            "Critical": "card-critical",
            "Moderate": "card-moderate",
            "Stable": "card-stable",
        }[risk_level]

        st.markdown(
            f"""
            <div class='card {color_class}'>
                <h4 style='margin:0;'>Risk Level: <span style='color:{risk_color(risk_level)}'>{risk_level}</span></h4>
                <p style='margin:0.4rem 0 0 0;'>Prediction Confidence: <b>{prob*100:.1f}%</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.metric("Risk Probability", f"{prob*100:.1f}%", delta=f"{proba_dict['Critical']*100:.1f}% critical likelihood")
        st.progress(min(max(int(prob * 100), 1), 100), text=f"Risk strength: {risk_level}")

        if risk_level == "Critical":
            st.error("🚨 Critical risk detected! Immediate clinician review recommended.")
            st.toast("Critical alert sent to caregiver dashboard", icon="🚨")
        elif risk_level == "Moderate":
            st.warning("⚠️ Moderate risk. Increase observation and reassess soon.")
            st.toast("Moderate alert logged", icon="⚠️")
        else:
            st.success("✅ Stable status at the moment.")
            st.toast("Patient marked stable", icon="✅")

        trend_df = forecast_24h_trend(int(heart_rate), int(spo2), float(temp), int(pain)).set_index("hour")
        st.markdown("<div class='section-title'>24-Hour Trend Prediction</div>", unsafe_allow_html=True)
        st.line_chart(trend_df)


elif page == "📊 Dashboard":
    st.markdown("<div class='section-title'>Caregiver Dashboard</div><div class='divider'></div>", unsafe_allow_html=True)

    if not st.session_state.logs:
        st.info("No patient logs yet. Add readings in Patient Monitoring to populate analytics.")
    else:
        logs_df = pd.DataFrame(st.session_state.logs)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Logs", len(logs_df))
        c2.metric("Latest Risk", logs_df.iloc[-1]["risk"])
        c3.metric("Avg Pain", f"{logs_df['pain'].mean():.1f}")

        if logs_df.iloc[-1]["risk"] == "Critical":
            st.markdown(
                "<div class='card card-critical'><b>🚨 Active Critical Alert:</b> Latest reading needs immediate attention.</div>",
                unsafe_allow_html=True,
            )

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown("<div class='card'><b>Pain Trend</b>", unsafe_allow_html=True)
            st.line_chart(logs_df.set_index("timestamp")[["pain"]])
            st.markdown("</div>", unsafe_allow_html=True)

        with chart_col2:
            st.markdown("<div class='card'><b>SpO2 Trend</b>", unsafe_allow_html=True)
            st.line_chart(logs_df.set_index("timestamp")[["spo2"]])
            st.markdown("</div>", unsafe_allow_html=True)

        risk_map = {"Stable": 0, "Moderate": 1, "Critical": 2}
        risk_hist = logs_df[["timestamp", "risk"]].copy()
        risk_hist["risk_value"] = risk_hist["risk"].map(risk_map)
        st.markdown("<div class='card'><b>Risk History</b>", unsafe_allow_html=True)
        st.line_chart(risk_hist.set_index("timestamp")[["risk_value"]])
        st.caption("Risk scale: 0 = Stable, 1 = Moderate, 2 = Critical")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Medication Reminder</div>", unsafe_allow_html=True)
    reminder = st.text_input("Add a reminder", placeholder="e.g., 8:00 PM - Morphine 5mg")
    if st.button("Save Reminder"):
        if reminder.strip():
            st.success(f"Reminder saved: {reminder}")
            st.toast("Medication reminder added", icon="💊")
        else:
            st.warning("Please enter a reminder first.")


else:  # Chatbot
    st.markdown("<div class='section-title'>Emotional Support Chatbot</div><div class='divider'></div>", unsafe_allow_html=True)

    chat_holder = st.container()
    with chat_holder:
        for msg in st.session_state.chat_history:
            css_class = "chat-user" if msg["role"] == "user" else "chat-assistant"
            st.markdown(f"<div class='{css_class}'>{msg['content']}</div>", unsafe_allow_html=True)

    user_msg = st.text_input("Share how you feel", placeholder="I'm feeling worried about tonight...")
    if st.button("Send"):
        if user_msg.strip():
            st.session_state.chat_history.append({"role": "user", "content": user_msg})
            emotion, response = generate_support_response(user_msg)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": f"Detected Emotion: **{emotion}**\n\n{response}"}
            )
            st.toast(f"Emotion detected: {emotion}", icon="💬")
            st.rerun()
        else:
            st.warning("Please type a message first.")


st.markdown("<div class='footer'>AI Palliative Care Assistant • Built for compassionate, data-driven care.</div>", unsafe_allow_html=True)
