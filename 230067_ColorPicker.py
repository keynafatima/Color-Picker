import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
from collections import Counter

st.set_page_config(page_title="Color Picker Gambar", layout="centered")
st.title("🎨 Color Picker Dominan dari Gambar")

uploaded_file = st.file_uploader("Unggah gambar (jpg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Gambar yang Diunggah", use_column_width=True)

    # Resize gambar
    image = image.resize((150, 150))
    img_array = np.array(image)
    pixels = img_array.reshape((-1, 3))

    # Clustering dengan MiniBatchKMeans
    k = 5
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(pixels)
    counts = Counter(labels)

    # Mengurutkan warna berdasarkan frekuensi
    center_colors = kmeans.cluster_centers_.astype(int)
    ordered = sorted(zip(counts.values(), center_colors), reverse=True)
    ordered_colors = [color for _, color in ordered]
    hex_colors = ['#{:02x}{:02x}{:02x}'.format(*c) for c in ordered_colors]

    # Menampilkan palet warna 
    st.subheader("Palet 5 Warna Dominan:")
    cols = st.columns(k)
    for i, col in enumerate(cols):
        col.color_picker(f"Warna #{i+1}", hex_colors[i], label_visibility="collapsed")
        col.markdown(f"<div style='text-align: center'>{hex_colors[i]}</div>", unsafe_allow_html=True)
