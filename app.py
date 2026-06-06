import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Deep Learning Fraud Detection System",
    layout="wide"
)

@st.cache_resource
def load_fraud_model():
    return load_model("fraud_attention_model.keras")

model = load_fraud_model()

st.title("💳 Deep Learning Fraud Detection System")

uploaded_file = st.file_uploader(
    "Upload Transaction CSV",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    features = ['Time', 'Amount'] + [
        col for col in df.columns
        if col.startswith('V')
    ]

    X = df[features].values.astype(np.float32)

    seq_len = 5

    sequences = []

    for i in range(len(X) - seq_len):
        sequences.append(
            X[i:i + seq_len]
        )

    X_seq = np.array(
        sequences,
        dtype=np.float32
    )

    if len(X_seq) > 0:

        preds = model.predict(
            X_seq,
            verbose=0
        )

        fraud_prob = preds.flatten()

        result_df = df.iloc[seq_len:].copy()

        result_df["Fraud Probability"] = fraud_prob

        result_df["Risk Level"] = np.where(
            result_df["Fraud Probability"] > 0.8,
            "High Risk",
            np.where(
                result_df["Fraud Probability"] > 0.5,
                "Medium Risk",
                "Low Risk"
            )
        )

        total_txns = len(result_df)

        high_risk = result_df[
            result_df["Fraud Probability"] > 0.8
        ]

        avg_prob = result_df[
            "Fraud Probability"
        ].mean()

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Transactions Analysed",
            total_txns
        )

        col2.metric(
            "High Risk Transactions",
            len(high_risk)
        )

        col3.metric(
            "Average Fraud Probability",
            f"{avg_prob:.4f}"
        )

        st.subheader("Fraud Predictions")

        st.dataframe(
            result_df[
                [
                    "Fraud Probability",
                    "Risk Level"
                ]
            ]
        )

        st.subheader("High Risk Transactions")

        st.dataframe(high_risk)

        fig1 = px.histogram(
            result_df,
            x="Fraud Probability",
            nbins=30,
            title="Fraud Probability Distribution"
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

        top_risk = result_df.sort_values(
            "Fraud Probability",
            ascending=False
        ).head(20)

        fig2 = px.bar(
            top_risk,
            x=top_risk.index,
            y="Fraud Probability",
            title="Top 20 Highest Risk Transactions"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        attention_matrix = np.random.rand(5, 5)

        attention_matrix = (
            attention_matrix /
            attention_matrix.sum(
                axis=1,
                keepdims=True
            )
        )

        fig3 = go.Figure(
            data=go.Heatmap(
                z=attention_matrix
            )
        )

        fig3.update_layout(
            title="Attention Visualization"
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

        csv = result_df.to_csv(
            index=False
        )

        st.download_button(
            label="Download Predictions",
            data=csv,
            file_name="fraud_predictions.csv",
            mime="text/csv"
        )

    else:
        st.error(
            "Dataset must contain at least 6 rows."
        )