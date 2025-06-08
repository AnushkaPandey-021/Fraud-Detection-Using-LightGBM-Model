import streamlit as st
import pandas as pd
import plotly.express as px
from Folder.load import predict_from_csv, expected_features
from Folder.preprocessing import preprocess_fraud_data
from streamlit_lottie import st_lottie
import json
import os
import sys

# --- Fix Module Import (Works from ANY location) ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# --- Cyberpunk Color Scheme ---
COLOR_SCHEME = {
    'background': '#0b0b0d',
    'primary': '#FF00FF',
    'accent': '#FF33CC',
    'highlight': '#DA70D6',
    'alert': '#FF1493',
    'text': '#FFFFFF',
    'card': '#1b1b1d',
    'pie_colors': ['#800080', '#8B008B', '#9932CC', '#DA70D6', '#FF00FF']
}

@st.cache_resource
def load_lottie_local(filepath: str):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except:
        return None

# --- Page Config ---
st.set_page_config(
    page_title="Real-Time Fraud Detection",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Global Styling ---
st.markdown(f"""
<style>
    .stApp {{
        background-color: {COLOR_SCHEME['background']};
        color: {COLOR_SCHEME['text']};
    }}
    h1 {{
        color: {COLOR_SCHEME['primary']} !important;
        font-size: 2rem !important;
        margin-bottom: 0.2rem !important;
        text-align: center;
        text-shadow: 0 0 5px #FF00FF, 0 0 10px #FF33CC, 0 0 20px #DA70D6;
    }}
    .metric-badge {{
        font-size: 0.8em;
        padding: 3px 8px;
        border-radius: 12px;
        background: {COLOR_SCHEME['highlight']}40;
        color: {COLOR_SCHEME['highlight']};
    }}
    .upload-label {{
        text-align: center;
        font-weight: bold;
        color: {COLOR_SCHEME['highlight']};
        margin-bottom: 0.2rem;
    }}
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown('<h1>Fraud Detection Analytical Dashboard</h1>', unsafe_allow_html=True)
    lottie_json = load_lottie_local(r"C:\Users\anapa\OneDrive\Desktop\trial 1\Animation.json")
    if lottie_json:
        st_lottie(lottie_json, speed=1, height=140, key="sidebar_anim")
    st.markdown('<p class="upload-label">Upload Transaction Data</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["csv"])

st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)

if uploaded_file is not None:
    # --- Load & Predict ---
    df = pd.read_csv(uploaded_file)
    processed_df = preprocess_fraud_data(df, expected_features=expected_features)
    prediction_df = predict_from_csv(uploaded_file)
    processed_df['fraud_prediction'] = prediction_df['fraud_prediction']
    processed_df['fraud_probability'] = prediction_df['fraud_probability']

    # --- Debug Info ---
    with st.expander("🔍 Model Debug Info"):
        st.json({
            "Model Expected": expected_features[:10] + ["…"],
            "Data Received": list(processed_df.columns),
            "First 5 Predictions": prediction_df['fraud_prediction'].head().tolist(),
            "Probability Range": [
                round(prediction_df['fraud_probability'].min(), 4),
                round(prediction_df['fraud_probability'].max(), 4)
            ]
        })

    # --- Top Metrics (5 columns) ---
    cols = st.columns(5)
    metrics = [
        ("Total Transactions", len(df), None),
        ("Fraud Cases", prediction_df['fraud_prediction'].sum(), f"{100 * prediction_df['fraud_prediction'].mean():.1f}%"),
        ("Avg Amount", f"${df['amount'].mean():,.2f}" if 'amount' in df.columns else "N/A", None),
    ]

    # Suspicious Volume Index (SVI) with meaningful badge
    if 'amount' in processed_df.columns:
        total_amt = processed_df['amount'].sum()
        top10_amt = processed_df.nlargest(int(len(processed_df)*0.1), 'amount')['amount'].sum()
        svi = top10_amt / total_amt
        metrics.append((
            "Suspicious Volume Index (SVI)",
            f"{svi*100:.2f}%",
            f"${top10_amt:,.0f} top-10% volume"
        ))

    # Average Fraud Probability
    avg_prob = processed_df['fraud_probability'].mean() * 100
    metrics.append(("Avg Fraud Prob", f"{avg_prob:.1f}%", None))

    # Render metrics
    for col, (label, value, badge) in zip(cols, metrics):
        with col:
            st.metric(label, value)
            if badge:
                st.markdown(f'<div class="metric-badge">{badge}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # --- Charts Layout ---
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)

    # 1) Fraud Probability Histogram
    with row1_col1:
        fig1 = px.histogram(
            processed_df,
            x='fraud_probability',
            nbins=20,
            title="Fraud Probability Distribution",
            color_discrete_sequence=[COLOR_SCHEME['primary']],
            height=320
        )
        st.plotly_chart(fig1, use_container_width=True)

    # 2) Transaction Type Pie
    with row1_col2:
        tp = processed_df['transaction_type'].value_counts(normalize=True).mul(100).reset_index()
        tp.columns = ['transaction_type','percentage']
        fig2 = px.pie(
            tp, names='transaction_type', values='percentage',
            title="Transaction Types Distribution",
            color_discrete_sequence=COLOR_SCHEME['pie_colors'],
            hole=0.4,
            height=320
        )
        fig2.update_layout(
            plot_bgcolor=COLOR_SCHEME['card'],
            paper_bgcolor=COLOR_SCHEME['background'],
            font_color=COLOR_SCHEME['text'],
            margin=dict(t=40, b=20)
        )
        fig2.update_traces(
            textinfo='percent+label',
            pull=[0.05]*len(tp),
            marker=dict(line=dict(color=COLOR_SCHEME['background'], width=2))
        )
        st.plotly_chart(fig2, use_container_width=True)

    # 3) Fraudulent Transaction Amounts Box Plot
    with row2_col1:
        fraud_amt = processed_df[processed_df['fraud_prediction'] == 1]
        fig3 = px.box(
            fraud_amt, y='amount',
            title="Fraudulent Transaction Amounts",
            color_discrete_sequence=[COLOR_SCHEME['alert']],
            height=320
        )
        st.plotly_chart(fig3, use_container_width=True)

    # 4) RWTV Bar Chart (custom purple scale)
    with row2_col2:
        rwtv = processed_df.groupby('transaction_type')['amount'].sum().reset_index(name='weighted_value')
        fig4 = px.bar(
            rwtv, x='transaction_type', y='weighted_value', color='weighted_value',
            color_continuous_scale=[
                [0.0, '#3f0d8f'],
                [0.5, '#7a3eb1'],
                [1.0, '#c77dff']
            ],
            title="RWTV by Transaction Type",
            height=320
        )
        fig4.update_layout(
            plot_bgcolor=COLOR_SCHEME['card'],
            paper_bgcolor=COLOR_SCHEME['background'],
            font_color=COLOR_SCHEME['text'],
            title_x=0.5,
            margin=dict(t=50, b=20),
            xaxis_title="Transaction Type",
            yaxis_title="Weighted Value",
            xaxis=dict(tickfont=dict(size=12)),
            yaxis=dict(tickfont=dict(size=12)),
            coloraxis_colorbar=dict(
                title={'text':'Value','font':{'color':COLOR_SCHEME['text']}},
                tickfont=dict(color=COLOR_SCHEME['text'])
            )
        )
        fig4.update_traces(marker_line_width=1.5, marker_line_color=COLOR_SCHEME['text'])
        st.plotly_chart(fig4, use_container_width=True)

    # --- Download Predictions as CSV ---
    csv_data = processed_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Predictions as CSV",
        csv_data,
        file_name='fraud_predictions.csv',
        mime='text/csv'
    )

else:
    st.info("Please upload a CSV file to begin analysis.")
    st.image(
        "https://via.placeholder.com/800x400.png?text=Upload+Transaction+Data+CSV",
        use_container_width=True
    )
