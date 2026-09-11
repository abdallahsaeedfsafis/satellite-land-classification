"""
Satellite Land Classification — Interactive Dashboard

A Streamlit application for exploring and testing the models developed in this
project: two CNNs (PyTorch, Keras) and two fine-tuned Vision Transformers
(PyTorch, Keras), all trained to classify satellite image tiles as agricultural
or non-agricultural land.
"""

import streamlit as st

st.set_page_config(
    page_title="Satellite Land Classification",
    layout="wide",
)

# Minimal custom styling: neutral, professional color palette, no decorative elements.
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            background-color: #f5f6f8;
            border-right: 1px solid #e0e2e6;
        }
        [data-testid="stSidebar"] h1 {
            font-size: 1.35rem;
            margin-bottom: 0.1rem;
        }
        .sidebar-subtitle {
            color: #6b7280;
            font-size: 0.85rem;
            margin-bottom: 1.2rem;
        }
        .sidebar-nav-label {
            font-size: 0.75rem;
            font-weight: 600;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

PAGES = ["Predict", "Model Comparison", "About"]

with st.sidebar:
    st.title("Satellite Land Classification")
    st.markdown(
        '<div class="sidebar-subtitle">CNN & Vision Transformer comparison</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sidebar-nav-label">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")

    st.divider()
    st.caption("Binary classification of satellite image tiles: agricultural vs. non-agricultural land.")

if page == "Predict":
    from pages_content import predict_page
    predict_page.render()
elif page == "Model Comparison":
    from pages_content import comparison_page
    comparison_page.render()
elif page == "About":
    from pages_content import about_page
    about_page.render()