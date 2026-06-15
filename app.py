import os, json, time
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image
import matplotlib.cm as cm_color

st.set_page_config(page_title="Tomato Leaf Disease Detector", page_icon="leaf", layout="centered")

# Preprocessing functions for the supported pretrained bases.
PREPROCESSORS = {
    "MobileNetV2": tf.keras.applications.mobilenet_v2.preprocess_input,
    "EfficientNetB0": tf.keras.applications.efficientnet.preprocess_input,
    "ResNet50": tf.keras.applications.resnet.preprocess_input,
}


@st.cache_resource
def load_assets():
    with open("deploy_meta.json") as f:
        meta = json.load(f)
    model = tf.keras.models.load_model(meta["model_file"])
    return model, meta


def prepare(pil_img, img_size):
    """Resize and return the image on the raw 0..255 scale.

    The deployed model has its preprocessing built into the graph, so the raw
    image is the correct input to model.predict.
    """
    img = pil_img.convert("RGB").resize((img_size[1], img_size[0]))
    return np.array(img).astype("float32")


def locate_last_conv(model):
    """Return (base_model_or_None, last_conv_name).

    For a transfer-learning model the pretrained base is a nested sub-model, so
    the last convolutional layer is not visible at the top level. This descends
    into the nested base when necessary.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, layers.Conv2D):
            return None, layer.name
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            for sub in reversed(layer.layers):
                if isinstance(sub, layers.Conv2D):
                    return layer, sub.name
    return None, None


def gradcam(model, image_0_255, meta, pred_index=None):
    """Gradient-weighted class activation mapping.

    Handles both a plain model (convolution at the top level) and a transfer
    model whose pretrained base is nested. For the nested case the base is run
    standalone on its expected (preprocessed) input and the shared classifier
    head is re-applied, so the explanation corresponds to the real prediction.
    """
    base, last_conv = locate_last_conv(model)
    if last_conv is None:
        return None
    arr = image_0_255[np.newaxis, ...].astype("float32")

    try:
        if base is None:
            _ = model(tf.zeros((1, *meta["img_size"], 3)))
            grad_model = models.Model(model.inputs,
                                      [model.get_layer(last_conv).output, model.outputs[0]])
            with tf.GradientTape() as tape:
                conv_out, preds = grad_model(arr)
                idx = int(tf.argmax(preds[0])) if pred_index is None else pred_index
                channel = preds[:, idx]
        else:
            cam_base = models.Model(base.inputs,
                                    [base.get_layer(last_conv).output, base.output])
            prep = PREPROCESSORS.get(meta.get("preprocess", ""), lambda z: z)
            head = [l for l in model.layers if l.name in ("head_gap", "head_drop", "head_out")]
            with tf.GradientTape() as tape:
                x = prep(tf.identity(arr))
                conv_out, base_out = cam_base(x)
                h = base_out
                for layer in head:
                    h = layer(h)
                preds = h
                idx = int(tf.argmax(preds[0])) if pred_index is None else pred_index
                channel = preds[:, idx]

        grads = tape.gradient(channel, conv_out)
        if grads is None:
            return None
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_out = conv_out[0]
        heatmap = tf.squeeze(conv_out @ pooled[..., tf.newaxis])
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
    except Exception:
        return None


def overlay(image_0_255, heatmap, alpha=0.4):
    h, w = image_0_255.shape[:2]
    hm = tf.image.resize(heatmap[..., np.newaxis], (h, w)).numpy().squeeze()
    colored = cm_color.jet(hm)[..., :3] * 255.0
    return np.clip(image_0_255 * (1 - alpha) + colored * alpha, 0, 255).astype("uint8")


st.title("Tomato leaf disease detector")
st.write("Upload a photograph of a single tomato leaf. The model predicts the disease, "
         "shows how confident it is, and highlights the part of the leaf it used.")

model, meta = load_assets()
class_names = meta["class_names"]

with st.expander("About this tool"):
    st.write(
        "This classifier recognises ten tomato leaf diseases and a healthy class. "
        "It was trained on the Tomato Leaf Disease dataset with a leakage-controlled "
        "split, and the deployed model is a fine-tuned ResNet50. The heat map below "
        "each prediction shows the regions of the leaf the model used, so you can "
        "check that it is reasoning about the symptoms rather than the background."
    )

uploaded = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    pil_img = Image.open(uploaded)
    arr = prepare(pil_img, meta["img_size"])

    t0 = time.time()
    probs = model.predict(arr[np.newaxis, ...], verbose=0)[0]
    latency_ms = (time.time() - t0) * 1000

    top_idx = int(np.argmax(probs))
    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_img, caption="uploaded leaf", use_container_width=True)
    with col2:
        st.metric("prediction", class_names[top_idx].replace("_", " "))
        st.metric("confidence", f"{probs[top_idx] * 100:.1f}%")
        st.caption(f"inference time: {latency_ms:.0f} ms")

    st.subheader("Class probabilities")
    order = probs.argsort()[::-1][:5]
    st.bar_chart({class_names[i].replace("_", " "): float(probs[i]) for i in order})

    st.subheader("Where the model looked (Grad-CAM)")
    hm = gradcam(model, arr, meta, pred_index=top_idx)
    if hm is not None:
        st.image(overlay(arr, hm), caption="warmer regions drove the prediction",
                 use_container_width=True)
    else:
        st.info("The explanation heat map is not available for this image.")

    st.warning("This tool is decision support, not a diagnosis. Confirm important "
               "cases with an agronomist, especially for field photographs.")
else:
    st.info("Awaiting an image upload.")
