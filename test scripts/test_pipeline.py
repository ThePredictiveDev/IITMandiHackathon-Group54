#!/usr/bin/env python3
"""
test_pipeline.py — End-to-end test of Planner, Retrieval, Writer, and Verifier pipeline.

This script hardcodes a test query, runs:
  1) plan_fetch (Planner Agent)
  2) retrieve   (Retrieval Layer Glue)
  3) stream_answer (Writer Agent) — retried until verified
  4) verify_solution (Verifier Agent)

and prints out each stage’s output, repeating the Writer+Verifier loop
until the Verifier returns "Yes" or a maximum of 5 attempts is reached.
"""

import json
from agents.planner_agent import plan_fetch
from index_tools_build_and_retrieve.retrieval import retrieve
from agents.writer_agent import stream_answer
from agents.verifier_agent import verify_solution
from stores_mem_and_cache.cache import get_cached, set_cached

def main():
    # 1) Hardcoded test query
    query = "Why is my Simulink Real-Time task missing data samples?"
    cached = get_cached(query)
    if cached:
        print("=== Served From Cache ===\n")
        print(cached["final_answer"])
        print("\n=== End Cached Response ===")
        return
    # 2) Planner Agent
    print("=== Planner Output ===")
    plan = plan_fetch(query)
    print("Public CoT:\n", plan["cot_public"].strip())
    print("\nRecommended k:", plan["fetch"]["k"])
    print("Keywords:", ", ".join(plan["fetch"]["keywords"]))

    # 3) Retrieval Layer
    print("\n=== Retrieved Chunks ===")
    chunks = retrieve(query)
    for i, c in enumerate(chunks, start=1):
        print(f"[{i}] id={c['chunk_id']}, score={c['match_score']:.3f}")
        print("    url:", c.get("source_url"))
        snippet = c["chunk_text"].replace("\n", " ")
        print("    snippet:", snippet[:80] + "…")
        print("    chunk CoT (public):", c.get("cot_public", "").strip())
    k = plan["fetch"]["k"]
    print(f"\nUsing top {k} chunks for writing.\n")

    # 4) Writer + Verifier loop
    max_attempts = 5
    final_answer = None
    verif = None

    for attempt in range(1, max_attempts + 1):
        print(f"\n=== Writer Attempt #{attempt} ===\n")
        # Stream the Writer’s answer and capture it
        full_answer = ""
        for token in stream_answer(query, plan, chunks[:k]):
            #print(token, end="", flush=True)
            full_answer += token
        print("=== Writer generation complete; now verifying… ===\n")

        # Verify the solution
        print("=== Verifier Result ===")
        verif = verify_solution(query, plan, full_answer)
        print(json.dumps(verif, indent=2))

        if verif.get("verdict") == "Yes":
            final_answer = full_answer
            print("\nVerifier approved the solution. Here is the final answer:\n")
            print(final_answer)
            break
        else:
            print(f"\nVerifier rejected (leniency {verif['leniency']}). Retrying writer...\n")

    # 5) Display the final outcome
    #if verif and verif.get("verdict") == "Yes":
     #   print("\n=== Final Verified Answer ===\n")
      #  print(final_answer)
    #else:
     #   print("\n=== Verification Failed After Attempts ===")
      #  print("Showing last generated answer:\n")
       # print(final_answer)

    cache_payload = {
        "plan_public": plan["cot_public"],
        "fetch": plan["fetch"],
        "keywords": plan["fetch"]["keywords"],
        "chunks": [
            {
                "chunk_id": c["chunk_id"],
                "score": c["match_score"],
                "url": c.get("source_url"),
                "cot_public": c.get("cot_public")
            }
            for c in chunks[: plan["fetch"]["k"]]
                ],
        "final_answer": final_answer,
        "verifier": verif

    }
    set_cached(query, cache_payload)

if __name__ == "__main__":
    main()
