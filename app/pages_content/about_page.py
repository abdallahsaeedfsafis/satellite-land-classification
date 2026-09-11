"""
About page: project overview, dataset details, methodology, and links.
"""

import streamlit as st


def render():
    st.title("About This Project")

    st.markdown(
        """
        ## Overview

        This project implements a land classification system for agricultural
        applications using satellite imagery. It was built as an independent,
        from-scratch implementation inspired by IBM's "AI Capstone Project with
        Deep Learning" course structure, covering data handling, CNN development,
        and Vision Transformer integration.

        **Task**: Binary classification of satellite image tiles as either
        agricultural or non-agricultural land.

        ## Dataset

        - **Source**: Satellite image tiles (64x64 pixels, RGB)
        - **Size**: 6,000 images, perfectly balanced across 2 classes (3,000 each)
        - **Split**: 70% train / 15% validation / 15% test (4,200 / 900 / 900 images)

        ## Methodology

        The project follows three stages:

        1. **Data Handling** — comparing memory-based and generator-based data
           loading approaches, and building a data augmentation pipeline.
        2. **CNN Development** — building and training equivalent CNN architectures
           in both PyTorch and Keras, and comparing their performance.
        3. **Vision Transformer Integration** — fine-tuning a pre-trained ViT
           (`google/vit-base-patch16-224`) in both frameworks, and comparing its
           performance against the CNN baselines.

        All models were evaluated on the same held-out test set of 900 images,
        ensuring every comparison in this project is fair and consistent.

        ## Key Result

        Lightweight CNNs trained from scratch (99.5-99.7% accuracy) outperformed
        a much larger fine-tuned Vision Transformer (97.2-97.4% accuracy) on this
        specific task — a useful, realistic illustration that larger pre-trained
        models are not automatically better, particularly when there is a
        significant domain gap between the pre-training data and the target task.

        ## Project Structure

        - `src/data_handling/` — data loading, augmentation, and dataset splitting
        - `src/models/` — model architectures, training, and evaluation scripts
        - `docs/` — detailed write-ups for each project stage
        - `app/` — this interactive dashboard

        ## Links
        """
    )

    st.link_button(
        "View source code on GitHub",
        "https://github.com/abdallahsaeedfsafis/satellite-land-classification",
    )