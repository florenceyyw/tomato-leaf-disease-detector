import os, json, time, glob, re
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image
import matplotlib.cm as cm_color

st.set_page_config(page_title="Tomato Leaf Disease Detector",
                   page_icon="🍃", layout="wide")

# ----------------------------------------------------------------- styling
st.markdown("""
<style>
/* tighten the default Streamlit padding */
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1100px; }

/* header band */
.app-header {
  background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 55%, #43a047 100%);
  border-radius: 16px; padding: 28px 34px; color: #ffffff;
  box-shadow: 0 6px 20px rgba(27,94,32,0.18);
  animation: fadeInDown 0.5s ease both;
}
.app-header h1 { color:#fff; font-size: 2.05rem; font-weight: 800; margin: 0 0 6px 0; letter-spacing:-0.5px; }
.app-header p  { color: rgba(255,255,255,0.92); font-size: 1.02rem; margin: 0; }

/* metric chips */
.chip-row { display:flex; gap: 14px; margin-top: 18px; flex-wrap: wrap; }
.chip {
  background: rgba(255,255,255,0.16); border: 1px solid rgba(255,255,255,0.28);
  border-radius: 10px; padding: 8px 16px; color:#fff; backdrop-filter: blur(4px);
}
.chip .v { font-size: 1.25rem; font-weight: 700; line-height:1.1; }
.chip .l { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.6px; opacity:0.88; }

/* result card */
.result-card {
  background:#ffffff; border:1px solid #e6ebe6; border-radius: 14px;
  padding: 20px 22px; box-shadow: 0 2px 10px rgba(0,0,0,0.04);
  animation: fadeIn 0.45s ease both;
}
.pred-label { font-size: 1.7rem; font-weight: 800; color:#1b5e20; margin: 2px 0 2px 0; line-height:1.15; }
.pred-sub   { font-size: 0.8rem; text-transform: uppercase; letter-spacing:0.6px; color:#7a857a; }

/* confidence bar */
.conf-wrap { background:#eef2ee; border-radius: 9px; height: 16px; overflow:hidden; margin-top:6px; }
.conf-fill { height:100%; border-radius: 9px; animation: growBar 0.7s ease both; }

/* section heading */
.section-h { font-size:1.05rem; font-weight:700; color:#243024; margin: 6px 0 2px 0; }

/* probability rows */
.prob-row { margin: 7px 0; }
.prob-top { display:flex; justify-content:space-between; font-size:0.86rem; color:#37423700; }
.prob-name { color:#2f3a2f; font-size:0.86rem; }
.prob-val  { color:#5b665b; font-size:0.86rem; font-variant-numeric: tabular-nums; }
.prob-track{ background:#eef2ee; border-radius:6px; height:10px; overflow:hidden; margin-top:2px; }
.prob-bar  { height:100%; border-radius:6px; background: linear-gradient(90deg,#43a047,#66bb6a); animation: growBar 0.6s ease both; }

@keyframes fadeIn { from{opacity:0} to{opacity:1} }
@keyframes fadeInDown { from{opacity:0; transform:translateY(-8px)} to{opacity:1; transform:none} }
@keyframes growBar { from{width:0} }

/* sample thumbnail buttons: make them look like cards */
div[data-testid="column"] div.stButton > button {
  width:100%; border:1px solid #dfe6df; border-radius:8px; background:#fafdfa;
  color:#2f3a2f; font-size:0.78rem; font-weight:600; padding:6px 4px; transition: all 0.15s ease;
}
div[data-testid="column"] div.stButton > button:hover {
  border-color:#43a047; background:#f0f8f0; color:#1b5e20; transform: translateY(-1px);
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------- model
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
    img = pil_img.convert("RGB").resize((img_size[1], img_size[0]))
    return np.array(img).astype("float32")

def locate_last_conv(model):
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

def pretty(name):
    return name.replace("_", " ").replace("Two-spotted", "two-spotted")

def conf_color(p):
    if p >= 0.80: return "#2e7d32"      # green
    if p >= 0.50: return "#f9a825"      # amber
    return "#e53935"                     # red

# discover sample images in samples/ (filename -> class via best match)
@st.cache_data
def find_samples(class_names):
    paths = sorted(glob.glob("samples/*"))
    paths = [p for p in paths if p.lower().endswith((".jpg", ".jpeg", ".png"))]
    out = []
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0].lower()
        # match to a class by normalised name overlap
        best, score = None, 0
        for c in class_names:
            cn = c.lower().replace("_", " ").replace("-", " ")
            toks = set(re.findall(r"[a-z]+", cn))
            stoks = set(re.findall(r"[a-z]+", stem.replace("_", " ")))
            ov = len(toks & stoks)
            if ov > score:
                best, score = c, ov
        out.append((p, best if best else os.path.basename(p)))
    return out

# ----------------------------------------------------------------- header
model, meta = load_assets()
class_names = meta["class_names"]

st.markdown(f"""
<div class="app-header">
  <h1>Tomato Leaf Disease Detector</h1>
  <p>Upload a tomato leaf photograph and the model identifies the disease, reports its confidence,
  and highlights the regions it used to decide.</p>
  <div class="chip-row">
    <div class="chip"><div class="v">95.6%</div><div class="l">Test accuracy</div></div>
    <div class="chip"><div class="v">{len(class_names)}</div><div class="l">Disease classes</div></div>
    <div class="chip"><div class="v">ResNet50</div><div class="l">Architecture</div></div>
    <div class="chip"><div class="v">Grad-CAM</div><div class="l">Explainable</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

st.write("")

with st.expander("About this tool and how to read it"):
    st.write(
        "This classifier recognises ten tomato leaf diseases and a healthy class. It was trained "
        "on the Tomato Leaf Disease dataset using a leakage-controlled split, so that augmented "
        "copies of the same leaf could not appear in both training and testing. The deployed model "
        "is a fine-tuned ResNet50. For every prediction the heat map shows which regions of the leaf "
        "the model used, so you can check that it is reasoning about the symptoms rather than the "
        "background. The tool is decision support, not a diagnosis."
    )

# ----------------------------------------------------------------- input
left, right = st.columns([1, 1], gap="large")
with left:
    st.markdown('<div class="section-h">1. Provide a leaf image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload your own photograph", type=["jpg", "jpeg", "png"],
                                label_visibility="collapsed")

with right:
    st.markdown('<div class="section-h">or pick a sample leaf</div>', unsafe_allow_html=True)
    samples = find_samples(class_names)
    if "sample_pick" not in st.session_state:
        st.session_state.sample_pick = None
    if "sample_truth" not in st.session_state:
        st.session_state.sample_truth = None
    if samples:
        st.caption("Pick a leaf without knowing its disease, then see whether the model gets it right.")
        ncol = 4
        rows = (len(samples) + ncol - 1) // ncol
        k = 0
        for _ in range(rows):
            cols = st.columns(ncol)
            for c in cols:
                if k >= len(samples):
                    break
                path, cls = samples[k]
                with c:
                    st.image(path, use_container_width=True)
                    if st.button(f"Sample {k + 1}", key=f"s{k}"):
                        st.session_state.sample_pick = path
                        st.session_state.sample_truth = cls
                k += 1
    else:
        st.caption("Sample images are not available in this deployment.")

# resolve the active image
active_img = None
active_truth = None
if uploaded is not None:
    active_img = Image.open(uploaded)
    st.session_state.sample_pick = None
    st.session_state.sample_truth = None
elif st.session_state.sample_pick:
    active_img = Image.open(st.session_state.sample_pick)
    active_truth = st.session_state.sample_truth

# ----------------------------------------------------------------- results
st.write("")
if active_img is not None:
    arr = prepare(active_img, meta["img_size"])
    t0 = time.time()
    probs = model.predict(arr[np.newaxis, ...], verbose=0)[0]
    latency_ms = (time.time() - t0) * 1000
    top_idx = int(np.argmax(probs))
    top_p = float(probs[top_idx])

    st.markdown('<div class="section-h">2. Result</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.image(active_img, caption="input leaf", use_container_width=True)
    with c2:
        st.markdown(f"""
        <div class="result-card">
          <div class="pred-sub">Prediction</div>
          <div class="pred-label">{pretty(class_names[top_idx])}</div>
          <div class="pred-sub" style="margin-top:10px;">Confidence</div>
          <div class="conf-wrap"><div class="conf-fill"
               style="width:{top_p*100:.1f}%; background:{conf_color(top_p)};"></div></div>
          <div style="text-align:right; font-weight:700; color:{conf_color(top_p)};
               font-variant-numeric:tabular-nums; margin-top:4px;">{top_p*100:.1f}%</div>
          <div style="color:#7a857a; font-size:0.78rem; margin-top:8px;">
               inference time {latency_ms:.0f} ms</div>
        </div>
        """, unsafe_allow_html=True)

        # Reveal block: only for sample leaves, shown AFTER the prediction so the
        # demonstration is genuine (the leaf's true label was hidden until now).
        if active_truth is not None:
            correct = (class_names[top_idx] == active_truth)
            if correct:
                st.markdown(f"""
                <div style="margin-top:12px; background:#e8f5e9; border:1px solid #a5d6a7;
                     border-radius:10px; padding:12px 16px; animation: fadeIn 0.5s ease both;">
                  <span style="color:#2e7d32; font-weight:700;">Correct.</span>
                  <span style="color:#33543a;"> This leaf was labelled
                  <b>{pretty(active_truth)}</b> in the dataset, and the model agreed.</span>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="margin-top:12px; background:#fff4e5; border:1px solid #ffcc80;
                     border-radius:10px; padding:12px 16px; animation: fadeIn 0.5s ease both;">
                  <span style="color:#e65100; font-weight:700;">Mismatch.</span>
                  <span style="color:#5a4632;"> This leaf was labelled
                  <b>{pretty(active_truth)}</b> in the dataset, but the model predicted
                  <b>{pretty(class_names[top_idx])}</b>. The heat map below shows what it focused on.</span>
                </div>""", unsafe_allow_html=True)

    # probability distribution (custom styled, top 5)
    st.write("")
    st.markdown('<div class="section-h">Class probabilities (top 5)</div>', unsafe_allow_html=True)
    order = probs.argsort()[::-1][:5]
    rows_html = ""
    for i in order:
        rows_html += f"""
        <div class="prob-row">
          <div class="prob-top"><span class="prob-name">{pretty(class_names[i])}</span>
          <span class="prob-val">{probs[i]*100:.1f}%</span></div>
          <div class="prob-track"><div class="prob-bar" style="width:{probs[i]*100:.1f}%;"></div></div>
        </div>"""
    st.markdown(rows_html, unsafe_allow_html=True)

    # grad-cam
    st.write("")
    st.markdown('<div class="section-h">3. Where the model looked (Grad-CAM)</div>', unsafe_allow_html=True)
    hm = gradcam(model, arr, meta, pred_index=top_idx)
    g1, g2 = st.columns([1, 1], gap="large")
    with g1:
        if hm is not None:
            st.image(overlay(arr, hm), caption="warmer regions drove the prediction",
                     use_container_width=True)
        else:
            st.info("The explanation heat map is not available for this image.")
    with g2:
        st.markdown(
            '<div style="color:#4a554a; font-size:0.9rem; padding-top:6px;">'
            'The heat map overlays the model\'s attention on the leaf. Warm regions (red and yellow) '
            'are those that most influenced the prediction. For a trustworthy result these should fall '
            'on the visible symptoms, the lesions and discoloured tissue, rather than on the background.'
            '</div>', unsafe_allow_html=True)

    st.write("")
    st.warning("This tool is decision support, not a diagnosis. Confirm important cases with an "
               "agronomist, especially for field photographs.")
else:
    st.info("Upload a leaf image or pick a sample above to see a prediction.")

# ----------------------------------------------------------------- project info
st.markdown("""
<div style="margin-top:40px; padding:18px 22px; border-top:1px solid #e6ebe6;
     color:#7a857a; font-size:0.82rem; line-height:1.65;">
  <b style="color:#33543a;">Sunway University &mdash; School of Mathematical Sciences</b><br>
  MAT5124 Machine Learning &middot; Group Project Report &middot; June 2026<br>
  <i>Tomato Leaf Disease Detection Using Convolutional Neural Networks, Transfer Learning,
  Explainable Artificial Intelligence, and Deployment</i><br>
  Project theme: Artificial Intelligence Solutions Lab, from Training to Real World Deployment<br>
  Prepared by Wong Yoke Yan (15093446), Yeong Jiun Shiung (25109521), Darren Yap Yee Shern (21001235)
</div>
""", unsafe_allow_html=True)
