# 🎮 Multimodal Game Recommendation System

An **AI-powered content-based video game recommendation system** that leverages **multimodal deep learning** (text + images) and **cross-modal retrieval using CLIP** to recommend similar games — even in **cold-start scenarios**.

---

## 📌 Project Overview

Traditional recommendation systems rely on user ratings or interaction history, which often fails for **new users** or **new games**.  
This project solves that problem by using **game content itself**:

- 📄 **Text** → Game description & genres  
- 🖼️ **Images** → Game posters  
- 🔀 **Multimodal embeddings** → Semantic similarity  
- 🔗 **CLIP** → Image-to-text recommendation  

The final system is deployed as an **interactive Streamlit web application**.

---

## 🧠 Key Features

- ✅ Content-based recommendations (no user data required)
- ✅ Multimodal learning (text + image)
- ✅ Cross-modal image → game search using CLIP
- ✅ Cold-start friendly
- ✅ Scalable embedding-based design
- ✅ Interactive Streamlit UI
- ✅ Production-ready deployment setup

---

## 🏗️ System Architecture
Stage 1: Prototype & Validation
└─ Small dataset to verify pipeline correctness

Stage 2: Multimodal Embedding Generation
├─ BERT → Text embeddings
├─ ResNet50 → Image embeddings
└─ Fusion → Final game embedding

Stage 3: Cross-Modal CLIP Retrieval
├─ CLIP Text Embeddings (offline)
└─ Image → Text similarity search

Deployment
└─ Streamlit + Render


---

## 🔍 Recommendation Modes

### 1️⃣ Text-Based Recommendation
- Select a game
- System finds similar games using **multimodal embeddings**
- Similarity measured using **cosine similarity**

### 2️⃣ Image-Based Recommendation (CLIP)
- Upload a game image
- CLIP encodes the image
- Matches it against CLIP text embeddings of games
- Returns visually and semantically similar games

---

## 🧪 Technologies Used

### 🔹 Core ML / DL
- **BERT** (text understanding)
- **ResNet50** (visual feature extraction)
- **CLIP (ViT-B/32)** for cross-modal learning
- **Transfer Learning** (frozen encoders)

### 🔹 Libraries
- `numpy`, `pandas`
- `scikit-learn`
- `torch`, `torchvision`
- `CLIP (OpenAI)`
- `streamlit`

### 🔹 Deployment
- **Render**
- Lightweight runtime (no TensorFlow required)

---

## 📁 Repository Structure

```text
├── app.py                         # Streamlit application
├── model.ipynb                    # Stage-2 embedding generation
├── build_clip_text_embeddings.py  # Stage-3 CLIP text embeddings
├── game_embeddings.npy            # Final multimodal embeddings
├── clip_text_embeddings.npy       # CLIP text embeddings
├── game_metadata.csv              # Clean metadata used by app
├── assets/
│   ├── banner.jpg                 # UI banner
│   └── demo.mp4                   # Demo video (added manually)
├── requirements.txt
├── .gitignore
└── README.md

▶️ Demo Video

🎥 Project Demo

assets/demo.mp4


You can:

Play it locally

Upload it to YouTube / Drive and link it here

Showcase it during interviews or presentations

🚀 How to Run Locally
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Run the App
streamlit run app.py

🧠 Design Decisions

Precomputed embeddings → Fast inference

No heavy ML libraries at runtime → Lightweight deployment

Cosine similarity → Ideal for high-dimensional embeddings

CLIP → Enables image-based recommendation without labels

Clean separation of stages → Industry-style ML system design

🎯 Interview One-Liner

“I built a multimodal content-based game recommendation system using pretrained BERT and ResNet50 embeddings, extended it with CLIP for cross-modal image-to-text retrieval, and deployed it as a scalable Streamlit application.”

👤 Author

Dhruv
IMSc – Quantitative Economics & Data Science
BIT Mesra

⭐ Final Note

This project demonstrates:

End-to-end ML system design

Multimodal deep learning

Transfer learning

Cross-modal retrieval

Deployment-ready engineering

If you find this project interesting, feel free to ⭐ the repository.



