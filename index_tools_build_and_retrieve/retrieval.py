"""
retrieval.py — “Planner-Validated” Retrieval Layer Glue

Workflow:
  retrieve(query):
    1) plan = plan_fetch(query)
    2) faiss.retrieve pool_size = floor(1.5 * k)
    3) build candidate list (truncated to 100 tokens each)
    4) scored = score_chunks(query, plan, candidates)
    5) return top k by match_score
"""

import math
import json
import faiss
from sentence_transformers import SentenceTransformer

from agents.planner_agent import plan_fetch, score_chunks

# ——————— Configuration ———————
FAISS_INDEX_PATH = "faiss.index"
CHUNKS_PATH      = "docs_chunks.jsonl"
EMBED_MODEL      = "intfloat/e5-small-v2"
EMBED_DEVICE     = "cuda" if faiss.get_num_gpus() > 0 else "cpu"
# ————————————————————————————

# 1) Load FAISS index
_index = faiss.read_index(r"data (json+index+raw csv)\faiss.index")

# 2) Load chunks in the same order used during indexing
_chunk_list = []
with open(r"data (json+index+raw csv)\docs_chunks.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        _chunk_list.append(rec)

# 3) Prepare embedder
_embedder = SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE)

def _embed_query(q: str):
    emb = _embedder.encode([q], convert_to_numpy=True, normalize_embeddings=True)
    return emb.astype("float32")

def retrieve(query: str):
    # 1) Plan fetch
    plan = plan_fetch(query)
    k = plan["fetch"]["k"]
    # 2) Determine pool size = floor(1.5 * k)
    pool_size = max(k, math.floor(1.5 * k))

    # 3) Embed and FAISS search
    q_emb = _embed_query(query)
    # for inner-product (cosine) indices, normalize
    if _index.metric_type == faiss.METRIC_INNER_PRODUCT:
        faiss.normalize_L2(q_emb)
    distances, indices = _index.search(q_emb, pool_size)

    # 4) Build candidate dicts (truncate to first 100 whitespace tokens)
    candidates = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx < 0 or idx >= len(_chunk_list):
            continue
        rec = _chunk_list[idx]
        words = rec["chunk_text"].split()
        truncated = " ".join(words[:100])
        candidates.append({
            "chunk_id":   rec["chunk_id"],
            "source_url": rec["source_url"],
            "chunk_text": truncated,
            "full_text":  rec["chunk_text"],
            "raw_score":  float(dist),
        })

    # 5) Ask Planner to semantically score & CoT each chunk
    scored = score_chunks(query, plan, candidates)

    # Merge metadata back into scored items
    id_map = {c['chunk_id']: c for c in candidates}
    for item in scored:
        cand = id_map[item['chunk_id']]
        item['source_url'] = cand['source_url']
        item['raw_score']  = cand['raw_score']
        # pass the full chunk text into the Writer
        item['chunk_text'] = cand['full_text']

    # 6) Return the top-k
    return scored