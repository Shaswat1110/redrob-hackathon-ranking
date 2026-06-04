# Redrob Hackathon: Intelligent Candidate Ranking System

This repository contains the code for the Decoupled Pre-computation Architecture to evaluate and rank candidate profiles for the Redrob Intelligent Candidate Discovery & Ranking Challenge. 

## Architecture Overview

The ranking system is designed to adhere strictly to the 5-minute CPU-only sandbox constraints while delivering state-of-the-art retrieval accuracy using a pre-computed dictionary lookup strategy.

1. **`build_manifest.py` (Offline Phase)**
   - Computes Hybrid Retrieval using `BM25` + `SentenceTransformers`.
   - Combines sparse and dense vectors via Reciprocal Rank Fusion (RRF).
   - Reranks top 1000 candidates with a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`).
   - Modulates scores based on behavioral heuristics from `redrob_signals`.
   - Generates reasoning justifications using local LLM inference (`Qwen/Qwen2.5-0.5B-Instruct`).
   - Exports the `precomputed_manifest.pkl`.

2. **`rank.py` (Runtime Phase)**
   - Sandbox-safe script (Zero ML dependencies).
   - Reads `candidates.jsonl` efficiently as a data stream ($O(1)$ memory usage).
   - Queries `precomputed_manifest.pkl` for pre-calculated AI scores.
   - Provides a fast, deterministic lexical overlap fallback for unknown candidates.
   - Outputs the exact `submission.csv` format required.

## Setup & Reproduction

### 1. Build the Offline Manifest
This step requires internet access to download model weights. It will parse `candidates.jsonl` and `job_description.md` from the parent directory to produce the `precomputed_manifest.pkl` file.

```bash
pip install -r requirements-offline.txt
python build_manifest.py
```

### 2. Generate the Submission CSV (The Sandbox Step)
This is the single command required by the hackathon submission spec to reproduce the ranking.

```bash
pip install -r requirements-runtime.txt
python rank.py --candidates ../candidates.jsonl --out submission.csv
```
