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

PAGES = ["Predict", "Model Comparison", "About"]

st.sidebar.title("Satellite Land Classification")
page = st.sidebar.radio("Navigation", PAGES)

if page == "Predict":
    from pages_content import predict_page
    predict_page.render()
elif page == "Model Comparison":
    from pages_content import comparison_page
    comparison_page.render()
elif page == "About":
    from pages_content import about_page
    about_page.render()