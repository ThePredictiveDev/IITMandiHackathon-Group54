#!/usr/bin/env python3
"""
test_retrieval_simple.py — Simple end-to-end retrieval + COT display
Hardcoded query, prints planner public reasoning and each chunk’s public COT.
"""

from agents.planner_agent import plan_fetch
from index_tools_build_and_retrieve.retrieval import retrieve

if __name__ == "__main__":
    # 1) Hardcoded test query
    query = "Why is my Simulink Real-Time task missing data samples?"

    # 2) Planner output
    plan = plan_fetch(query)
    k = plan["fetch"]["k"]

    print("\n=== Planner’s Public Reasoning ===")
    print(plan["cot_public"].strip())
    print(f"\n→ Planner recommends fetching k = {k} chunks")
    print("→ Keywords:", ", ".join(plan["fetch"]["keywords"]))

    # 3) Retrieval + reranking
    chunks = retrieve(query)[:k]

    # 4) Display each chunk’s public reasoning
    print(f"\n=== Top {k} Retrieved Chunks ===")
    for i, c in enumerate(chunks, start=1):
        print(f"\n[{i}] Chunk ID: {c['chunk_id']} (score={c['match_score']:.3f})")
        print("Source URL:", c.get("source_url"))
        print("Reasoning (Public):")
        print(c["cot_public"].strip())
        print("-" * 60)
