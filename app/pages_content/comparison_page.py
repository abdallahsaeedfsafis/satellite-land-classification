"""
Model Comparison page: displays the evaluation results of all four trained
models side by side, using the metrics documented in docs/module2_cnn_development.md
and docs/module3_vit_integration.md.
"""

import pandas as pd
import streamlit as st

RESULTS = [
    {
        "Model": "CNN (PyTorch)",
        "Accuracy": 0.9967,
        "Precision": 0.9977,
        "Recall": 0.9955,
        "F1-score": 0.9966,
        "Misclassifications": 3,
        "Parameters": "1,142,210",
    },
    {
        "Model": "CNN (Keras)",
        "Accuracy": 0.9956,
        "Precision": 0.9932,
        "Recall": 0.9977,
        "F1-score": 0.9955,
        "Misclassifications": 4,
        "Parameters": "1,142,210",
    },
    {
        "Model": "Vision Transformer (PyTorch)",
        "Accuracy": 0.9722,
        "Precision": 0.9770,
        "Recall": 0.9659,
        "F1-score": 0.9714,
        "Misclassifications": 25,
        "Parameters": "85,800,194 (1,538 trainable)",
    },
    {
        "Model": "Vision Transformer (Keras)",
        "Accuracy": 0.9744,
        "Precision": 0.9771,
        "Recall": 0.9705,
        "F1-score": 0.9738,
        "Misclassifications": 23,
        "Parameters": "85,800,194 (1,538 trainable)",
    },
]


def render():
    st.title("Model Comparison")
    st.write(
        "All four models were evaluated on the same held-out test set of 900 images, "
        "ensuring a fair, apples-to-apples comparison."
    )

    df = pd.DataFrame(RESULTS)
    display_df = df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1-score"]:
        display_df[col] = (display_df[col] * 100).round(2).astype(str) + "%"

    st.subheader("Test Set Metrics")
    st.dataframe(display_df, hide_index=True, use_container_width=True)

    st.subheader("Accuracy Comparison")
    chart_df = df.set_index("Model")[["Accuracy"]]
    st.bar_chart(chart_df, height=350)

    st.subheader("Key Findings")
    st.markdown(
        """
        - Both CNN models (PyTorch and Keras) achieved the highest accuracy, above 99.5%,
          with only 3-4 misclassifications out of 900 test samples.
        - Both Vision Transformer models achieved solid but lower accuracy, around 97.2-97.4%,
          despite having roughly 75x more total parameters than the CNNs.
        - The CNNs were trained from scratch specifically on this dataset, while the ViTs
          were pre-trained on natural images (ImageNet) and only had their classification
          head fine-tuned — this domain gap explains the performance difference.
        - Framework choice (PyTorch vs. Keras) had minimal impact on final accuracy for
          either architecture, confirming the comparison is primarily about CNN vs. ViT,
          not about framework quality.
        """
    )
    st.caption("See docs/module2_cnn_development.md and docs/module3_vit_integration.md for full details.")