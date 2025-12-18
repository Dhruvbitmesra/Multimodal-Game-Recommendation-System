import streamlit as st
import numpy as np
import pandas as pd
import os
import base64
import torch
import clip
from PIL import Image

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# =================================================
# CONFIG
# =================================================
st.set_page_config(
    page_title="🎮 Multimodal Game Recommendation System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

IMAGE_DIR = "images"
BANNER_PATH = "assets/gta.jpg"

# =================================================
# SESSION STATE
# =================================================
if "top_k" not in st.session_state:
    st.session_state.top_k = 5

# =================================================
# UTILS
# =================================================
def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

banner_base64 = img_to_base64(BANNER_PATH)

# =================================================
# LOAD DATA (INDEX SAFE)
# =================================================
@st.cache_data
def load_data():
    df = pd.read_csv("game_metadata.csv").reset_index(drop=True)
    text_emb = normalize(np.load("game_embeddings.npy"))
    clip_text_emb = normalize(np.load("clip_text_embeddings.npy"))
    return df, text_emb, clip_text_emb

df, text_embeddings, clip_text_embeddings = load_data()

# =================================================
# LOAD CLIP
# =================================================
@st.cache_resource
def load_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess, device

clip_model, clip_preprocess, clip_device = load_clip()

def get_image_embedding(path):
    image = Image.open(path).convert("RGB")
    image = clip_preprocess(image).unsqueeze(0).to(clip_device)
    with torch.no_grad():
        emb = clip_model.encode_image(image)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()

# =================================================
# GLOBAL CSS (FINAL – NO REGRESSIONS)
# =================================================
st.markdown("""
<style>

/* === REMOVE STREAMLIT CHROME === */
header, footer { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }

/* REMOVE EXTRA PADDING */
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 0rem !important;
}

/* === BACKGROUND GRADIENT === */
.stApp {
    background: linear-gradient(
        135deg,
        #ff9a3c,
        #ffb347,
        #7f7cff,
        #4facfe
    );
    background-size: 400% 400%;
    animation: gradientMove 18s ease infinite;
    overflow-x: hidden;
}

@keyframes gradientMove {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* === FLOATING GAME ICONS (RESTORED) === */
.floating-icons span {
    position: fixed;
    bottom: -10%;
    font-size: 28px;
    opacity: 0.15;
    animation: floatUp linear infinite;
    z-index: 0;
}

.floating-icons span:nth-child(1){left:10%;animation-duration:22s;}
.floating-icons span:nth-child(2){left:25%;animation-duration:26s;}
.floating-icons span:nth-child(3){left:45%;animation-duration:20s;}
.floating-icons span:nth-child(4){left:65%;animation-duration:28s;}
.floating-icons span:nth-child(5){left:85%;animation-duration:24s;}

@keyframes floatUp {
    from { transform: translateY(0); }
    to { transform: translateY(-140vh); }
}

/* === WAVES === */
.wave {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 200%;
    height: 220px;
    background: rgba(255,255,255,0.25);
    border-radius: 100% 100% 0 0;
    animation: waveMove 14s linear infinite;
    z-index: 0;
}

.wave.wave2 {
    bottom: -30px;
    opacity: 0.18;
    animation-duration: 20s;
}

@keyframes waveMove {
    0% { transform: translateX(0); }
    100% { transform: translateX(-50%); }
}

/* === HERO === */
.hero-banner {
    height: 180px;
    border-radius: 18px;
    overflow: hidden;
    box-shadow: 0 0 30px rgba(0,0,0,0.25);
    margin-bottom: 16px;
}

.hero-banner img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

.hero-title {
    text-align: center;
    font-size: 2.6rem;
    font-weight: 900;
    color: white;
}

.hero-sub {
    text-align: center;
    color: rgba(255,255,255,0.9);
    margin-bottom: 22px;
}

/* === CARDS === */
.card {
    background: rgba(255,255,255,0.25);
    backdrop-filter: blur(14px);
    border-radius: 22px;
    padding: 18px;
    box-shadow: 0 12px 40px rgba(0,0,0,0.18);
}

/* === BAR === */
.bar {
    height: 10px;
    background: rgba(255,255,255,0.35);
    border-radius: 999px;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    background: linear-gradient(90deg,#ff9a3c,#4facfe);
}

</style>

<div class="floating-icons">
    <span>🎮</span><span>🕹️</span><span>👾</span><span>⚔️</span><span>🏎️</span>
</div>

<div class="wave"></div>
<div class="wave wave2"></div>
""", unsafe_allow_html=True)

# =================================================
# HERO
# =================================================
st.markdown(f"""
<div class="hero-banner">
    <img src="data:image/jpg;base64,{banner_base64}">
</div>
<div class="hero-title">🎮 Multimodal Game Recommendation System</div>
<div class="hero-sub">AI-powered game discovery using text & visual understanding</div>
""", unsafe_allow_html=True)

# =================================================
# MODE + TOP K
# =================================================
mode = st.radio("Recommendation mode", ["Text mode", "Image mode"], horizontal=True)

c1, c2, c3 = st.columns([1,6,1])
with c1:
    if st.button("➖"):
        st.session_state.top_k = max(3, st.session_state.top_k - 1)
with c3:
    if st.button("➕"):
        st.session_state.top_k = min(10, st.session_state.top_k + 1)

fill = int((st.session_state.top_k - 3) / 7 * 100)
st.markdown(f"""
<div class="bar"><div class="bar-fill" style="width:{fill}%"></div></div>
<p style="text-align:center;color:white;">Top {st.session_state.top_k} recommendations</p>
""", unsafe_allow_html=True)

# =================================================
# SEARCH + SELECT
# =================================================
l, r = st.columns(2)
with l:
    query = st.text_input("🔍 Search game")
with r:
    names = df["name"]
    if query:
        names = names[names.str.contains(query, case=False, na=False)]
    selected_game = st.selectbox("📂 Select game", names.tolist())

game_row = df[df["name"] == selected_game].iloc[0]
game_idx = df.index[df["name"] == selected_game][0]

# =================================================
# SELECTED GAME CARD
# =================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
c1, c2 = st.columns([1,2])
with c1:
    img = f"{IMAGE_DIR}/{game_row['appid']}.jpg"
    if os.path.exists(img):
        st.image(img, use_container_width=True)
with c2:
    st.markdown(f"## {game_row['name']}")
st.markdown('</div>', unsafe_allow_html=True)

# =================================================
# IMAGE UPLOAD
# =================================================
image_emb = None
if mode == "Image mode":
    uploaded = st.file_uploader("Upload game image", ["jpg","jpeg","png"])
    if uploaded:
        path = f"{IMAGE_DIR}/query.png"
        with open(path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.image(path, use_container_width=True)
        image_emb = get_image_embedding(path)

# =================================================
# RECOMMENDERS
# =================================================
def recommend_text(idx, k):
    sims = cosine_similarity(
        text_embeddings[idx].reshape(1,-1),
        text_embeddings
    )[0]
    ids = sims.argsort()[::-1][1:k+1]
    res = df.iloc[ids].copy()
    res["sim"] = sims[ids]
    return res

def recommend_image(img_emb, k):
    sims = cosine_similarity(img_emb, clip_text_embeddings)[0]
    ids = sims.argsort()[::-1][:k]
    res = df.iloc[ids].copy()
    res["sim"] = sims[ids]
    return res

recs = (
    recommend_image(image_emb, st.session_state.top_k)
    if mode == "Image mode" and image_emb is not None
    else recommend_text(game_idx, st.session_state.top_k)
)

# =================================================
# RESULTS
# =================================================
cols = st.columns(len(recs))
for col, (_, row) in zip(cols, recs.iterrows()):
    with col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        img = f"{IMAGE_DIR}/{row['appid']}.jpg"
        if os.path.exists(img):
            st.image(img, use_container_width=True)
        pct = int(row["sim"] * 100)
        st.markdown(f"""
        <div class="bar">
            <div class="bar-fill" style="width:{pct}%"></div>
        </div>
        <p style="text-align:center;color:white;">{pct}% match</p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
