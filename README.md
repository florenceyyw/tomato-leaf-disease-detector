---
title: Tomato Leaf Disease Detector
emoji: 🍅
colorFrom: green
colorTo: red
sdk: docker
app_port: 8501
pinned: false
license: mit
---

# Tomato Leaf Disease Detector

An interactive web application that classifies tomato leaf diseases from a photograph
of a single leaf. The model recognises ten diseases and a healthy class, shows its
confidence, and displays a gradient-weighted class activation (Grad-CAM) heat map so
the user can see which part of the leaf drove the prediction.

The deployed model is a fine-tuned ResNet50 trained on the Tomato Leaf Disease dataset
with a leakage-controlled train, validation, and test split. It reached approximately
95.6 percent accuracy on a held-out test set.

## Components

- **app.py** — the Streamlit web interface (this is what the Space runs).
- **api.py** — an optional REST API built with FastAPI, which exposes the same model
  programmatically.

## Running the web interface locally

    pip install -r requirements.txt
    streamlit run app.py

## Running the REST API locally

    uvicorn api:app --host 0.0.0.0 --port 8000

Then:

- `GET  http://localhost:8000/` returns a health check and metadata.
- `GET  http://localhost:8000/classes` returns the list of disease classes.
- `POST http://localhost:8000/predict` accepts an image file and returns the predicted
  class, the confidence, and the full probability distribution as JSON. Interactive API
  documentation is available at `http://localhost:8000/docs`.

Built for the MAT5124 Machine Learning group project.
