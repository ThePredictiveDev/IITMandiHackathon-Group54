#!/usr/bin/env python3
"""
planner_agent.py — Planner Agent (LLM-A) using Groq client

Uses Groq's Python SDK to call llama3-8b-8192 for planning fetches.
"""

import json
import time
from groq import Groq

# Instantiate Groq client
API_KEY="" #YOUR_API_KEY_HERE
client = Groq(api_key=API_KEY)

# Few-shot system prompt
SYSTEM_PROMPT = """You are a Planner Agent for a MATLAB Simulink Real-Time troubleshooting assistant.
Your job: Given a user query (Check if query related to MATLAB, Simulink, or any MathWorks product) and maybe some user chat memory (long term + short term), estimate how many document chunks (k) are needed and extract the most salient
domain-specific keywords. Produce ONLY valid JSON with keys:
- "cot_raw": the full chain-of-thought reasoning (private for system use) end line with the marker: <<END_COT>>.
- "cot_public": a concise, user-facing summary of reasoning end line with the marker: <<END_COT>>.
- "fetch": an object with:
    - "k": integer, estimated number of chunks to retrieve.
    - "keywords": list of 3–7 lowercase keywords or phrases, sorted by importance.

Rules:
1. Use sophisticated logic to pick keywords: focus on technical terms and error-specific phrases.
2. Estimate k based on query complexity (short/simple → 4, medium → 6, complex → 8+).
3. Do not output any additional keys or explanatory text.
4. If your JSON is malformed, output EXACTLY the text: INVALID_JSON
5. After your final reasoning line (cot_raw and cot_public), output the marker: <<END_COT>>.
6. Then on the next line, output your JSON.
7. If the query is not about MATLAB, Simulink, or any MathWorks product, output:
    {
    "query": <Query>,
    "response": "QUERY NOT RELATED"
    }

"""

# Few-shot examples
EXAMPLES = [
    {
        "query": "After changing the Receive block's Sample time to 0.5, my Scope output distorts the sine wave. What's the cause?",
        "response": {
            "cot_raw": "The user describes a mismatch between send and receive rates: Send block at sample 0.1, Receive at 0.5. This will drop messages in the queue leading to distortion. Key signals: Sample time, queue drop. <<END_COT>>",
            "cot_public": "There’s a sampling rate mismatch: you set receive to 0.5 while messages come at 0.1, causing skipped data. <<END_COT>>",
            "fetch": {
                "k": 6,
                "keywords": ["sample time", "receive block", "queue capacity", "message drop"]
            }
        }
    },
    {
        "query": "Why am I getting ‘Overloaded CPU’ warnings when executing my Real-Time C code on the target?",
        "response": {
            "cot_raw": "The warning indicates the target CPU cannot keep up with real-time tasks. Possible issues: C code inefficiency, high sample rates, missing optimization flags. <<END_COT>>",
            "cot_public": "Your CPU is overburdened by too-fast execution or unoptimized code causing real-time overruns. <<END_COT>>",
            "fetch": {
                "k": 5,
                "keywords": ["overloaded cpu", "real-time c", "optimization", "sample rate"]
            }
        }
    },

    {
    "query": "How do I make pancakes?",
    "response": "QUERY NOT RELATED"
    },

    {
    "query": "I keep getting a 'License checkout failed' error when launching Simulink on startup. What's causing this and how can I fix it?",
    "response": {
        "cot_raw": "A license checkout failure usually means Simulink cannot reach the license server or the license file is invalid or expired. Possible causes: network issues preventing server contact, incorrect license path in MATLAB settings, or expired/renewed license not updated on the client. Check network connectivity to the license host, verify the license file location under 'Help → Licensing → Manage License', and ensure your license is current. <<END_COT>>",
        "cot_public": "Simulink is failing to check out a license because it either can't reach the license server, the license file path is wrong, or the license has expired. Check your network connection, licensing settings, and license validity. <<END_COT>>",
        "fetch": {
            "k": 5,
            "keywords": [
                "license checkout failed",
                "Simulink startup",
                "license server",
                "license file path",
                "network issues"
            ]
        }
    }
    },

    {
    "query": "Can you recommend a good Italian restaurant near me?",
    "response": "QUERY NOT RELATED"
    }


]

def build_messages(query: str):
    """
    Assemble system+examples+user for Groq chat API.
    """
    messages = []
    messages.append({"role": "system", "content": SYSTEM_PROMPT})
    for ex in EXAMPLES:
        messages.append({"role": "user", "content": ex["query"]})
        messages.append({
            "role": "assistant",
            "content": json.dumps(ex["response"], ensure_ascii=False)
        })
    messages.append({"role": "user", "content": query})
    return messages

def plan_fetch(query: str, max_retries=2) -> dict:
    """
    Call Groq chat completion to plan fetch parameters.
    Retries on INVALID_JSON or malformed output.
    """
    messages = build_messages(query)
    for attempt in range(1, max_retries+1):
        comp = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages,
            temperature=0.0,
            top_p=1.0,
            max_completion_tokens=512,
            stream=False,
            stop=None
        )
        text = comp.choices[0].message.content.strip()
        # Handle off-topic queries
        # 1) Try parsing JSON to catch the off-topic response format
        try:
            parsed = json.loads(text)
            # If it matches our off-topic schema, short-circuit
            if isinstance(parsed, dict) and parsed.get("response") == "QUERY NOT RELATED":
                return "QUERY NOT RELATED"
        except json.JSONDecodeError:
            # not JSON or empty—fall through to normal handling
            pass
        
        if text == "INVALID_JSON":
            # Ask for strict JSON
            messages.append({
                "role": "assistant",
                "content": "Your last output was INVALID_JSON. Please output only valid JSON following the schema."
            })
            continue
        try:
            result = json.loads(text)
            # validate keys
            assert "cot_raw" in result and "cot_public" in result and "fetch" in result
            fetch = result["fetch"]
            assert isinstance(fetch.get("k"), int)
            assert isinstance(fetch.get("keywords"), list)
            return result
        except Exception:
            if attempt < max_retries:
                messages.append({
                    "role": "assistant",
                    "content": "Your last output was invalid JSON or missing required fields. Please respond with only the JSON schema."
                })
                continue
            raise RuntimeError(f"Planner failed after {max_retries} attempts; last output: {text}")
    # unreachable
    return {}

def score_chunks(query: str, plan: dict, chunks: list[dict], max_retries: int = 2) -> list[dict]:
    """
    Re-rank candidate chunks semantically via the Planner LLM.
    Args:
      query: the original user question
      plan:  the dict returned by plan_fetch (with cot_raw, cot_public, fetch)
      chunks: list of dicts, each with keys:
        - chunk_id, source_url, chunk_text (first 100 tokens), token_count
    Returns:
      A list of up to plan['fetch']['k'] dicts, each:
        {
          "chunk_id": str,
          "match_score": float,      # 0.0–1.0
          "cot_raw": str,            # private chain-of-thought
          "cot_public": str          # user-facing rationale
        }
      sorted by descending match_score.
    """
    # Build the messages for scoring
    SYSTEM = f"""You are a Planner Agent for MATLAB Simulink Real-Time troubleshooting.
Your task now: given the original query and plan, evaluate each candidate chunk for relevance.
Use the plan’s keywords to guide your reasoning. Output a full chain-of-thought (private),
then output ONLY valid JSON: a list of objects with keys:
- "chunk_id"
- "match_score" (float 0.0–1.0)
- "cot_raw"
- "cot_public"

Rules:
1. After your private reasoning, output the JSON exactly, no extra text.
2. If you produce malformed JSON, output EXACTLY INVALID_JSON.
3. Return only the top {plan['fetch']['k']} chunks by match_score.
"""
    
    # ─── Examples for chunk scoring ───
    EXAMPLES_SCORE = [
        {
            "chunks": [
                {
                    "chunk_id": "ac0dc552407dc206b80d65v73e8aeef0_12",
                    "source_url": "https://in.mathworks.com/help/slrealtime/ug/",
                    "chunk_text": "The buffer is too small causing overflow and data loss."
                },
                {
                    "chunk_id": "ac0dc552407dc983b80d65d69e8acdf2_12",
                    "source_url": "https://in.mathworks.com/help/slrealtime/ug/",
                    "chunk_text": "Adjust sample time to match send and receive rates to avoid skipped samples."
                }
            ],
            "response": [
                {
                    "chunk_id": "ac0dc552407dc206b80d65v73e8aeef0_12",
                    "match_score": 0.95,
                    "cot_raw": "Chunk c1 directly addresses buffer overflow and matches the plan’s keywords about buffer size and overflow. <<END_COT>>",
                    "cot_public": "This chunk explains how buffer overflow causes data loss. <<END_COT>>"
                },
                {
                    "chunk_id": "ac0dc552407dc983b80d65d69e8acdf2_12",
                    "match_score": 0.60,
                    "cot_raw": "Chunk c2 is about sample time mismatch, which is related but less critical than buffer issues. <<END_COT>>",
                    "cot_public": "This chunk discusses sample time adjustments. <<END_COT>>"
                }
            ]
        }
    ] 
    # assemble messages
    messages = [{"role":"system","content":SYSTEM}]
    # ─── Few‐shot examples for chunk scoring ───
    for ex in EXAMPLES_SCORE:
        messages.append({
            "role": "user",
            "content": "Example candidate chunks:\n" + 
                       json.dumps(ex["chunks"], ensure_ascii=False, indent=2)
        })
        messages.append({
            "role": "assistant",
            "content": json.dumps(ex["response"], ensure_ascii=False, indent=2)
        })
    # include the original plan for context
    messages.append({"role":"assistant","content":json.dumps(plan, ensure_ascii=False)})
    # list each chunk (id + truncated text)
    chunks_list = []
    for c in chunks:
        chunks_list.append({
            "chunk_id": c["chunk_id"],
            "source_url": c["source_url"],
            "chunk_text": c["chunk_text"]
        })
    messages.append({"role":"user","content":
        "Here are the candidate chunks (first 100 tokens each):\n" +
        json.dumps(chunks_list, ensure_ascii=False, indent=2)
    })
    # ask for scoring
    for attempt in range(1, max_retries+1):
        comp = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages,
            temperature=0.0,
            top_p=1.0,
            max_completion_tokens=1024,
            stream=False,
            stop=None
        )
        out = comp.choices[0].message.content.strip()
        if out == "INVALID_JSON":
            if attempt < max_retries:
                messages.append({"role":"assistant","content":"INVALID_JSON"})
                continue
            raise RuntimeError("Score_chunks: INVALID_JSON twice")
        try:
            scored = json.loads(out)
            # ensure we have the right schema
            assert isinstance(scored, list)
            for item in scored:
                assert "chunk_id" in item and "match_score" in item
                assert "cot_raw" in item and "cot_public" in item
            # sort & trim
            scored = sorted(scored, key=lambda x: x["match_score"], reverse=True)
            return scored[: plan["fetch"]["k"]]
        except Exception:
            if attempt < max_retries:
                messages.append({"role":"assistant","content":"INVALID_JSON"})
                continue
            raise


if __name__ == "__main__":
    q = "Why is my Simulink Real-Time task missing data samples?"
    plan = plan_fetch(q)
    print(json.dumps(plan, indent=2, ensure_ascii=False))
