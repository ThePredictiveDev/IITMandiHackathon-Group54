#!/usr/bin/env python3
import argparse, json, faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def load_metadata(path):
    meta = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            meta.append(json.loads(line))
    return meta

def embed_query(query, model, device):
    model = SentenceTransformer(model, device=device)
    q_emb = model.encode([query], normalize_embeddings=True)
    return q_emb

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index",   required=True, help="faiss.index")
    p.add_argument("--meta",    required=True, help="metadata.jsonl")
    p.add_argument("--model",   default="intfloat/e5-small-v2")
    p.add_argument("--device",  default=None)
    p.add_argument("--k",       type=int, default=5, help="top-k results")
    p.add_argument("--query",   required=True, help="text query")
    args = p.parse_args()

    device = args.device or ("cuda" if faiss.get_num_gpus()>0 else "cpu")
    print(f"Using device: {device}")

    # 1) load
    print("Loading FAISS index…")
    index = faiss.read_index(args.index)

    print("Loading metadata…")
    metadata = load_metadata(args.meta)

    # 2) embed
    print(f"Embedding query: {args.query}")
    q_emb = embed_query(args.query, args.model, device).astype("float32")

    # 3) search
    print(f"Searching top {args.k}…")
    scores, ids = index.search(q_emb, args.k)

    # 4) display
    seen_urls = set()
    count = 0
    for rank, (idx, score) in enumerate(zip(ids[0], scores[0]), start=1):
        m = metadata[idx]
        url = m["source_url"]
        if url in seen_urls:
            continue
        seen_urls.add(url)
        snippet = m.get("chunk_text", "")[:100].replace("\n"," ") + "…"
        print(f"{count+1}. [{score:.4f}] {url}")
        print(f"Tags: {m['tags']}")
        print(f"Snippet: {snippet}\n")
        count += 1
        if count >= args.k:
            break

if __name__=="__main__":
    main()
