import clip
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
CSV_PATH = "game_metadata.csv"
OUTPUT_PATH = "clip_text_embeddings.npy"
MODEL_NAME = "ViT-B/32"
MAX_TOKENS = 77  # CLIP hard limit

# -----------------------------
# DEVICE
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# -----------------------------
# LOAD MODEL
# -----------------------------
model, _ = clip.load(MODEL_NAME, device=device)
model.eval()

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(CSV_PATH)

print(f"Building CLIP text embeddings for {len(df)} games")

# -----------------------------
# TEXT BUILDER
# -----------------------------
def build_clip_text(row):
    name = str(row.get("name", ""))

    genres = row.get("genres", "")
    if not isinstance(genres, str):
        genres = ""

    description = row.get("about_the_game", "")
    if not isinstance(description, str):
        description = ""

    # Keep description short on purpose
    description = description[:200]

    text = (
        f"{name}. "
        f"Genres: {genres}. "
        f"{description}"
    )

    return text.strip()

# -----------------------------
# SAFE TOKENIZATION
# -----------------------------
def tokenize_safe(text):
    tokens = clip.tokenize([text], truncate=True)
    return tokens[:, :MAX_TOKENS]

# -----------------------------
# EMBEDDING LOOP
# -----------------------------
embeddings = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    text = build_clip_text(row)

    tokens = tokenize_safe(text).to(device)

    with torch.no_grad():
        emb = model.encode_text(tokens)

    emb = emb / emb.norm(dim=-1, keepdim=True)
    embeddings.append(emb.cpu().numpy()[0])

# -----------------------------
# SAVE
# -----------------------------
embeddings = np.vstack(embeddings)
np.save(OUTPUT_PATH, embeddings)

print("✅ CLIP text embeddings created successfully!")
print("Saved to:", OUTPUT_PATH)
print("Shape:", embeddings.shape)
