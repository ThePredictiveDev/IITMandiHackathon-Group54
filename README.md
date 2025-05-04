# MATLAB / Simulink Real‑Time Troubleshooter – Project Overview

This repository contains a complete multi‑agent Retrieval‑Augmented Generation (RAG) stack that ingests MathWorks documentation and powers an interactive troubleshooting assistant with full chain‑of‑thought transparency.

---

## 📂 Script Glossary

| Script | Purpose (one‑liner) |
|--------|--------------------|
| prep_matlab_docs.py | Cleans the scraped CSV, de‑duplicates sentences, extracts tags, and emits cleaned pages as JSONL. |
| chunk_docs.py | Tokenises, window‑chunks and strides the cleaned pages to 256‑token segments (32‑token stride). |
| build_index.py | Embeds chunks with the E5‑small‑v2 model and builds a FAISS‑HNSW index plus an embeddings cache. |
| planner_agent.py | LLM‑powered planner that selects k, keywords, rejects off‑topic queries, and re‑scores candidate chunks. |
| retrieval.py | Hybrid layer combining FAISS recall with planner validation to return full‑length chunks. |
| writer_agent.py | Streams THOUGHT / ACTION / EVIDENCE answers with numbered MathWorks citations. |
| verifier_agent.py | Lightweight verifier that votes Yes or No on Writer output with adaptive leniency and retry logic. |
| memory.py | Dual‑tier chat memory: short‑term list plus long‑term Redis store with TTL eviction. |
| cache.py | Hot‑path Redis cache (15‑minute TTL) storing final user‑facing answers only. |
| chatbot.py | End‑to‑end orchestrator for a single turn, including memory queries, caching, planning, retrieval, writing and verification. |
| frontend.py | Gradio UI with chat on the left and collapsible panels for Memory, full Chain‑of‑Thought and Detailed Logs on the right. |
| batch_chatbot_demo.py | Runs fifteen preset Simulink questions through the pipeline and logs Writer outputs to a text file. |
| test_memory.py | Diagnostics to visualise STM / LTM contents and confirm TTL‑based eviction. |

---

## ▶️ Running the Assistant (CLI)

1. Start a local Redis instance.  
2. Build the FAISS index once with build_index.py (chunks, embeddings, metadata).  
3. Launch chatbot.py and type MATLAB / Simulink questions in the terminal.  
   The script prints Planner, Chunk, Verifier and Writer chains‑of‑thought.

---

## 🌐 Running the Gradio Front‑End

Execute frontend.py.  
A browser tab opens with a ChatGPT‑style interface.  
Conversation streams on the left; Memory, full CoT and Detailed Logs are collapsible on the right.  
Passing share=True to demo.launch provides a public link.

---

## 📝 Batch Evaluation

Running batch_chatbot_demo.py executes fifteen diverse Simulink questions and saves each Writer THOUGHT / ACTION / EVIDENCE triple to batch_results.txt for offline review.

---

All scripts share common artefacts (faiss.index, metadata.jsonl, embeddings.npy) and environment variables (Groq API key, Redis address). Once the index is built, you can switch seamlessly between CLI, Gradio or batch modes without re‑processing documentation.
