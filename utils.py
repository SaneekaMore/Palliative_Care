"""Shared utility helpers for UI state and formatting."""

from __future__ import annotations

from datetime import datetime


def init_session_state() -> None:
    """Initialize Streamlit session-state containers."""
    import streamlit as st

    defaults = {
        "logs": [],
        "risk_history": [],
        "chat_history": [
            {
                "role": "assistant",
                "content": "Hello, I'm your support companion. How are you feeling today?",
            }
        ],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def risk_color(risk_level: str) -> str:
    """Map risk labels to semantic colors."""
    return {
        "Critical": "#D32F2F",
        "Moderate": "#F57C00",
        "Stable": "#2E7D32",
    }.get(risk_level, "#2E7D32")


def append_log(heart_rate: int, spo2: int, temp: float, pain: int, risk_level: str, probability: float) -> dict:
    """Build a new patient log entry."""
    return {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "heart_rate": heart_rate,
        "spo2": spo2,
        "temp": temp,
        "pain": pain,
        "risk": risk_level,
        "probability": round(probability * 100, 1),
    }


def render_css() -> None:
    """Inject custom CSS for modern healthcare UI."""
    import streamlit as st

    st.markdown(
        """
        <style>
            .stApp {
                background: linear-gradient(180deg, #E8F5E9 0%, #F5FBF6 100%);
                color: #1F2A1F;
            }
            .main-header {
                background: #4CAF50;
                color: white;
                border-radius: 16px;
                padding: 1.2rem 1.5rem;
                margin-bottom: 1rem;
                box-shadow: 0 6px 18px rgba(0,0,0,0.08);
            }
            .card {
                background: white;
                border-radius: 16px;
                padding: 1.1rem 1.2rem;
                margin: 0.65rem 0;
                border: 1px solid #D9EEDB;
                box-shadow: 0 4px 14px rgba(76,175,80,0.08);
            }
            .card-critical { border-left: 8px solid #D32F2F; }
            .card-moderate { border-left: 8px solid #F57C00; }
            .card-stable { border-left: 8px solid #2E7D32; }
            .section-title {
                font-size: 1.08rem;
                font-weight: 700;
                margin-top: 0.75rem;
                margin-bottom: 0.35rem;
                color: #285C2C;
            }
            .divider {
                border-top: 1px solid #D7E9D9;
                margin: 0.4rem 0 0.9rem 0;
            }
            .chat-user, .chat-assistant {
                border-radius: 14px;
                padding: 0.65rem 0.8rem;
                margin: 0.3rem 0;
                max-width: 90%;
            }
            .chat-user {
                background: #DDF4DE;
                margin-left: auto;
                text-align: right;
            }
            .chat-assistant {
                background: #FFFFFF;
                border: 1px solid #D9EEDB;
                margin-right: auto;
            }
            .footer {
                text-align: center;
                color: #5F7B63;
                margin-top: 2rem;
                font-size: 0.86rem;
            }
            .stButton > button {
                background: #4CAF50;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 0.5rem 0.9rem;
                transition: all 0.2s ease;
            }
            .stButton > button:hover {
                background: #43A047;
                transform: translateY(-1px);
            }
            .stNumberInput input, .stTextInput input {
                border-radius: 10px !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
