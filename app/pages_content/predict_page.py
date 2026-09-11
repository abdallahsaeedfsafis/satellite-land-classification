"""
Predict page: upload a satellite image tile and get a prediction from any
of the four trained models (CNN/ViT, PyTorch/Keras).
"""

import os
import sys

import numpy as np
import streamlit as st
import torch
from PIL import Image

# Make src/models and src/data_handling importable from the app.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(PROJECT_ROOT, "src", "models"))
sys.path.append(os.path.join(PROJECT_ROOT, "src", "data_handling"))

CLASS_LABELS = {0: "Non-Agricultural Land", 1: "Agricultural Land"}

MODEL_OPTIONS = [
    "CNN (PyTorch)",
    "CNN (Keras)",
    "Vision Transformer (PyTorch)",
]


@st.cache_resource
def load_cnn_pytorch():
    from cnn_pytorch import SimpleCNN
    model = SimpleCNN(num_classes=2)
    path = os.path.join(PROJECT_ROOT, "outputs", "models", "cnn_pytorch.pt")
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


@st.cache_resource
def load_cnn_keras():
    import tensorflow as tf
    path = os.path.join(PROJECT_ROOT, "outputs", "models", "cnn_keras.keras")
    return tf.keras.models.load_model(path)


@st.cache_resource
def load_vit_pytorch():
    from vit_pytorch import build_vit, get_image_processor
    model = build_vit(num_classes=2)
    path = os.path.join(PROJECT_ROOT, "outputs", "models", "vit_pytorch.pt")
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    processor = get_image_processor()
    return model, processor




def predict_cnn_pytorch(image: Image.Image):
    from torchvision import transforms
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    tensor = transform(image).unsqueeze(0)

    model = load_cnn_pytorch()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).numpy()[0]
    return probs


def predict_cnn_keras(image: Image.Image):
    resized = image.resize((64, 64))
    array = np.array(resized, dtype="float32") / 255.0
    array = np.expand_dims(array, axis=0)

    model = load_cnn_keras()
    logits = model.predict(array, verbose=0)[0]
    probs = np.exp(logits) / np.sum(np.exp(logits))  # softmax
    return probs


def predict_vit_pytorch(image: Image.Image):
    model, processor = load_vit_pytorch()
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = model(pixel_values=inputs["pixel_values"]).logits
        probs = torch.softmax(logits, dim=1).numpy()[0]
    return probs



PREDICT_FUNCTIONS = {
    "CNN (PyTorch)": predict_cnn_pytorch,
    "CNN (Keras)": predict_cnn_keras,
    "Vision Transformer (PyTorch)": predict_vit_pytorch,
}


def render():
    st.title("Land Classification — Prediction")
    st.caption("Upload a satellite image tile and classify it using one of the four trained models.")
    st.divider()

    col1, col2 = st.columns([1, 1])

    with col1:
        model_choice = st.selectbox("Select model", MODEL_OPTIONS)
        uploaded_file = st.file_uploader(
            "Upload an image (JPG or PNG)", type=["jpg", "jpeg", "png"]
        )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        with col1:
            st.image(image, caption="Uploaded image", width=250)
            run_prediction = st.button("Run Prediction", type="primary")

        with col2:
            if run_prediction:
                with st.spinner("Running inference..."):
                    probs = PREDICT_FUNCTIONS[model_choice](image)

                predicted_idx = int(np.argmax(probs))
                predicted_label = CLASS_LABELS[predicted_idx]
                confidence = float(probs[predicted_idx])

                st.subheader("Result")
                st.metric("Predicted Class", predicted_label)
                st.metric("Confidence", f"{confidence * 100:.2f}%")

                st.subheader("Class Probabilities")
                st.bar_chart(
                    {"Probability": [probs[0], probs[1]]},
                    x_label="Class",
                    height=250,
                )
                st.caption(
                    f"Non-Agricultural: {probs[0] * 100:.2f}%  |  "
                    f"Agricultural: {probs[1] * 100:.2f}%"
                )
    else:
        st.info("Upload an image to get a prediction.")