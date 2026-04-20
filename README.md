# AI-Powered Palliative Care Assistant System

A modern, visually polished Streamlit application for palliative care workflows. It combines patient risk monitoring, caregiver analytics, and an emotional support chatbot in one clean interface.

## Features

- **🩺 Patient Monitoring & Risk Prediction**
  - Inputs: Heart Rate, SpO2, Temperature, Pain Level
  - ML model: `RandomForestClassifier`
  - Outputs: risk class (**Stable / Moderate / Critical**), confidence score, and 24-hour trend forecast
- **💬 Emotional Support Chatbot**
  - Sentiment analysis powered by TextBlob
  - Emotion categories: **Happy / Neutral / Sad / Anxious**
  - Supportive and calming contextual responses
- **📊 Caregiver Dashboard**
  - In-memory logging of monitoring events
  - Pain, SpO2, and risk trend visualizations
  - Critical warning banner and medication reminder input
- **🎨 Enhanced UI/UX**
  - Custom CSS with healthcare palette (`#4CAF50`, `#E8F5E9`)
  - Rounded cards, spacing, subtle shadows, icon-based navigation
  - Toast notifications, metrics, progress visualization, and structured sections

## Tech Stack

- Python
- Streamlit
- scikit-learn
- pandas
- numpy
- TextBlob

## Project Structure

- `app.py` – Main Streamlit UI and navigation
- `model.py` – Synthetic data generation + model training + forecasting
- `chatbot.py` – Sentiment-driven emotional support logic
- `utils.py` – CSS, state initialization, shared helpers
- `requirements.txt` – Dependencies

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown in the terminal (typically `http://localhost:8501`).

## Alert Rules

- **Critical:** `SpO2 < 90` OR `Pain > 8`
- **Moderate:** Mild deviations (e.g., low-normal oxygen, elevated pain/temp/HR)
- **Stable:** Otherwise

## Screenshots

> Add screenshots after launching the app locally.

Suggested captures:
1. Patient Monitoring page with risk card
2. Dashboard with trend charts
3. Chatbot conversation view
