# Redrob Hackathon: Intelligent Candidate Ranking System

**👉 [Live Sandbox Demo (Google Colab)](https://colab.research.google.com/drive/15Nj25tDUI71cQSngzW6CjCTUD0x_Zk90?usp=sharing)**

This repository contains the code for our **Decoupled Pre-computation Architecture**, designed to evaluate and rank 100,000 candidate profiles for the Redrob Intelligent Candidate Discovery & Ranking Challenge. 

## 🏆 Architecture Philosophy

The ranking system was engineered to deliver **State-of-the-Art (SOTA) retrieval accuracy** while strictly adhering to the **5-minute CPU-only sandbox constraints**. 

To achieve this, we decoupled the architecture into two distinct engines: a heavy Offline ML Builder, and an ultra-lightweight Runtime Ranker.

### 1. The Offline Engine (`build_manifest.py`)
This script performs the heavy ML lifting without runtime constraints.
- **Hybrid Retrieval:** Computes `BM25` (sparse) and `all-MiniLM-L6-v2` (dense) embeddings.
- **Reciprocal Rank Fusion (RRF):** Merges sparse and dense vectors to find the Top 1000 candidates.
- **Cross-Encoder Re-ranking:** Re-ranks the Top 1000 using `ms-marco-MiniLM-L-6-v2` for highly precise semantic matching.
- **Behavioral Modifiers:** Adjusts logits based on `redrob_signals` (e.g., rewarding GitHub activity, penalizing long notice periods).
- **Honeypot Handling:** Intercepts integrity honeypots and forces their scores to `-9999`.
- **LLM Reasoning:** Generates unique, non-templated reasoning justifications using local LLM inference (`Qwen/Qwen2.5-0.5B-Instruct`).
- **Caching:** Exports all results and states to `precomputed_manifest.pkl`.

### 2. The Sandbox Engine (`rank.py`)
This is the script executed in the Hackathon Sandbox.
- **Zero ML Dependencies:** Built entirely with `pandas` and standard Python libraries to ensure maximum compatibility and speed.
- **$O(1)$ Memory Streaming:** Reads the massive `candidates.jsonl` file line-by-line, easily bypassing the 16GB RAM limit.
- **Mathematical Scaling:** Uses a custom Min-Max Scaler to cleanly format raw Cross-Encoder logits into `[0.8000, 0.9999]` probabilities.
- **Performance:** Finishes evaluating 100,000 candidates in **<10 seconds** on a standard CPU.

---

## 🚀 Setup & Reproduction

### Step 1: Build the Offline Manifest
*Note: This step requires internet access to download model weights. We have already included the generated `.pkl` file in this repository so this step can be bypassed.*

```bash
pip install -r requirements-offline.txt
python build_manifest.py
```

### Step 2: Generate the Submission CSV (The Sandbox Step)
This is the single command required by the hackathon submission spec to reproduce the ranking inside the CPU-only sandbox.

```bash
pip install -r requirements-runtime.txt
python rank.py --candidates ../candidates.jsonl --out submission.csv
```
