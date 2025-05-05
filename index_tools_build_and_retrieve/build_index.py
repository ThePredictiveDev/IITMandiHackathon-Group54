#!/usr/bin/env python3
"""
build_index.py — Embed chunks + build FAISS HNSW index (improved)

Usage example:
  python build_index.py \
    --chunks   docs_chunks.jsonl \
    --index    faiss.index \
    --meta     metadata.jsonl \
    --cache    embeddings.npy \
    --model    intfloat/e5-small-v2 \
    --batch-size 32 \
    --M        32 \
    --ef-constr 64 \
    --ef-search 128 \
    --metric   ip \
    --device   cuda \
    --mmap-base faiss_mmap \
    --seed     42
"""

import argparse, json, logging, sys, time
from pathlib import Path

import numpy as np

try:
    import faiss
except ImportError:
    raise ImportError("Please install faiss-cpu or faiss-gpu")

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ─────────────────── Logging ───────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)

# ─────────────────── Helpers ───────────────────
def load_chunks(path):
    texts, meta = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            texts.append(rec["chunk_text"])
            meta.append({
                "chunk_id":   rec["chunk_id"],
                "source_url": rec["source_url"],
                "title":      rec.get("title", ""),
                "tags":       rec.get("tags", []),
                "chunk_text": rec["chunk_text"]
            })
    return texts, meta

def embed_chunks(texts, model_name, bs, device, cache_path=None):
    if cache_path and Path(cache_path).exists():
        logging.info(f"Loading embeddings from cache: {cache_path}")
        return np.load(cache_path)
    logging.info(f"Embedding {len(texts)} chunks with {model_name} on {device}")
    model = SentenceTransformer(model_name, device=device)
    embs = model.encode(
        texts,
        batch_size=bs,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    if cache_path:
        np.save(cache_path, embs)
        logging.info(f"Saved embeddings to cache: {cache_path}")
    return embs

def build_hnsw_index(dim, M, efC, metric, seed=None):
    if seed is not None:
        np.random.seed(seed)
    metric_id = faiss.METRIC_INNER_PRODUCT if metric=="ip" else faiss.METRIC_L2
    idx = faiss.IndexHNSWFlat(dim, M, metric_id)
    idx.hnsw.efConstruction = efC
    return idx

def add_batches(idx, embs, bs):
    for i in tqdm(range(0, len(embs), bs), desc="Adding to FAISS"):
        idx.add(embs[i:i+bs])
    return idx

def to_gpu(idx, requested_device):
    ng = faiss.get_num_gpus()
    if requested_device.startswith("cuda") and ng>0:
        logging.info(f"Offloading index to {ng} GPU(s)")
        return faiss.index_cpu_to_all_gpus(idx)
    if requested_device.startswith("cuda"):
        logging.warning("Requested CUDA but no GPUs detected; using CPU index")
    return idx

def write_metadata(meta, out):
    with open(out, "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False)+"\n")

# ─────────────────── Main ───────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chunks",    required=True)
    p.add_argument("--index",     required=True)
    p.add_argument("--meta",      required=True)
    p.add_argument("--cache",     default=None)
    p.add_argument("--mmap-base", default=None)
    p.add_argument("--model",     default="intfloat/e5-small-v2")
    p.add_argument("--batch-size",type=int, default=32)
    p.add_argument("--M",         type=int, default=32)
    p.add_argument("--ef-constr", type=int, default=64)
    p.add_argument("--ef-search", type=int, default=128)
    p.add_argument("--metric",    choices=["ip","l2"], default="ip")
    p.add_argument("--device",    default=None)
    p.add_argument("--seed",      type=int, default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if faiss.get_num_gpus()>0 else "cpu")
    logging.info(f"Running on device: {device}")

    # 1) Load data
    texts, meta = load_chunks(args.chunks)
    # 2) Embed
    embs = embed_chunks(texts, args.model, args.batch_size, device, cache_path=args.cache)
    dim  = embs.shape[1]
    logging.info(f"Embeddings shape: {embs.shape}")

    # 3) Build index
    idx = build_hnsw_index(dim, args.M, args.ef_constr, args.metric, seed=args.seed)
    idx = to_gpu(idx, device)

    # 4) Add in batches
    idx = add_batches(idx, embs, args.batch_size)
    idx.hnsw.efSearch = args.ef_search

    # 5) Persist
    if args.mmap_base:
        logging.info(f"Writing index → {args.index} and mmap → {args.mmap_base}.cmap")
        faiss.write_index(idx, args.index)
        try:
            faiss.write_index_mmap(idx, args.mmap_base)
        except AttributeError:
            logging.warning("write_index_mmap unavailable; mmap skipped")
    else:
        logging.info(f"Writing index → {args.index}")
        faiss.write_index(idx, args.index)

    # 6) Metadata
    logging.info(f"Writing metadata → {args.meta}")
    write_metadata(meta, args.meta)

    logging.info("✅ Done!")

if __name__=="__main__":
    main()
