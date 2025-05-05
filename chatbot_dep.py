#!/usr/bin/env python3
"""
chatbot.py — Orchestrator that prints only the four CoT stages:
  1. Planner CoT
  2. Planner‐Validated per‐chunk CoTs
  3. Verifier CoT (reason)
  4. Writer CoT
"""

import logging
from stores_mem_and_cache.memory import add_to_memory, get_memory
from agents.planner_agent import plan_fetch
from index_tools_build_and_retrieve.retrieval import retrieve
from agents.writer_agent import stream_answer
from agents.verifier_agent import verify_solution
from langdetect import detect
import re
from stores_mem_and_cache.cache import get_cached, set_cached
import json


# Match “what … last … message|say|chat” with any text in between
MEMORY_QUERY_RE = re.compile(
    r"""
    \bwhat\b.*\b(?:message|say|chat)\b.*\blast\b     # what … say … last
    |
    \bwhat\b.*\blast\b.*\b(?:message|say|chat)\b     # what … last … say
    """,
    re.I|re.X)
DOMAIN_KWS = re.compile(
    r"\b(error|sample|buffer|log|scope|queue|task|license|cpu|signal|troubleshoot|real[- ]?time|model|simulink|matlab)\b",
    re.I)

REMIND_RE = re.compile(r"\b(remind me|could you remind me)\b.*\b(what|which)\b.*\b(asked|said|message)\b", re.I)

TROUBLE_KWS = re.compile(r"\b(error|why|how|fix|troubleshoot|issue|problem)\b", re.I)

MAX_INPUT_CHARS = 800

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mem_prefix = get_memory()
# Build a short “context” string:
stm_text = "\n".join(f"{t['role']}: {t['content']}" for t in mem_prefix["stm"])
ltm_text = "\n".join(f"Memory: {t['content']}" for t in mem_prefix["ltm"])
context_prefix = (
    "### Conversation history (last few turns):\n"
    + stm_text
    + (("\n### Longer‐term memories:\n" + ltm_text) if ltm_text else "")
    + "\n### Now, new user query:\n"
)


# ADD once near other helpers (adapted identical version)
def _writer_sections(txt: str) -> dict[str, str]:
    spec = {"THOUGHT": "COT", "ACTION": "ACTION", "EVIDENCE": "EVIDENCE"}
    out = {}
    for lab, end in spec.items():
        m = re.search(rf"<<{lab}>>(.*?)(?:<<END_{end}>>|$)", txt, re.S)
        blk = m.group(1).strip() if m else ""
        blk = re.sub(rf"^{lab}\s*:\s*", "", blk, flags=re.I)
        out[lab] = blk
    return out


def run_chat_turn(user_input: str):

    if not user_input or not user_input.strip():
        reply = "Please type your MATLAB/Simulink question."
        add_to_memory("assistant", reply)
        print("\nAssistant:\n", reply)
        return {"type":"memory", "message": reply}
    
    if len(user_input) > MAX_INPUT_CHARS:
        user_input = user_input[:MAX_INPUT_CHARS] + "…"
        print(f"YOUR INPUT WAS TRUNCATED AS IT WAS TOO LONG, TRUNCATION AT THE CHAR AT POSITION {MAX_INPUT_CHARS}")
        # (optionally inform the user)

    # 1) record user
    add_to_memory("user", user_input)

    # ————————————————
    # Quick memory‐query intents (no Planner!)
    # — Memory‐query intents: generic vs. domain‐specific —
    if MEMORY_QUERY_RE.search(user_input) or REMIND_RE.search(user_input):
        mem = get_memory()

        # a) Domain‐specific: “What was my last message about X?”
        if DOMAIN_KWS.search(user_input):
            # find the first domain keyword mentioned
            topic = DOMAIN_KWS.search(user_input).group(1).lower()
            # Gather all past user turns (most recent first), skip the current prompt
            user_turns = [t for t in reversed(mem["stm"]) if t["role"] == "user"][1:]
            # scan for the first matching topic
            for turn in user_turns:
                if topic in turn["content"].lower():
                    msg = re.sub(r"^You:\s*", "", turn["content"])
                    reply = f"Your most recent message about '{topic}' was:\n- {msg}"
                    break
            else:
                reply = f"I couldn't find any message about '{topic}' in recent history."

        # b) Generic: “What was my last message?”
        else:
            # gather last 3 user messages (excluding this prompt)
            last_users = [
                t["content"]
                for t in reversed(mem["stm"])
                if t["role"] == "user"
            ][1:4]
            if not last_users:
                reply = "I don't have a record of any previous messages yet."
            else:
                reply = "Your recent messages were:\n" + "\n".join(f"- {m}" for m in last_users)

        add_to_memory("assistant", reply)
        print("\nAssistant:\n", reply)
        return {"type": "memory", "message": reply}
    
    # ── Hot-path cache check ──
    cached = get_cached(user_input)
    if cached is not None:
        print(cached["display"])
        return {"type":"pipeline_cached", "message": cached["display"]}
    
    # try:
    #     if detect(user_input) != "en":
    #         reply = "Please ask your question in English."
    #         add_to_memory("assistant", reply)
    #         print("\nAssistant:\n", reply)
    #         return {"type":"not_english", "message": reply}
    # except Exception:
    #     # if detection fails, proceed anyway
    #     pass

    # 2) planner
    try:
        plan = plan_fetch(user_input)
    except Exception as e:
        logger.error(f"Planner error: {e}")
        reply = "Sorry, I couldn't understand your request. Please rephrase (PLEASE ONLY ASK MATLAB/SIMULINK RELATED QUESTIONS)."
        add_to_memory("assistant", reply)
        print("\nAssistant:\n", reply)
        return {"type":"error", "message": reply}
    # If planner says off-topic, return a friendly message
    if plan == "QUERY NOT RELATED":
        reply = "Sorry, I can only help with MATLAB/Simulink troubleshooting."
        add_to_memory("assistant", "Sorry, I can only help with MATLAB/Simulink troubleshooting.")
        print("\nAssistant:\nSorry, I can only help with MATLAB/Simulink troubleshooting.")
        return {"type":"off_topic", "message": reply}
    planner_cot = plan["cot_raw"]

    # 3) retrieval  per-chunk planner COT (we already have chunk_public)
    chunks = retrieve(user_input)
    if not chunks:
        reply = "I couldn't find any relevant MATLAB docs. Could you rephrase or provide more detail?"
        add_to_memory("assistant", reply)
        print("\nAssistant:\n", reply)
        return {"type":"error", "message": reply}

    k = plan["fetch"]["k"]
    topk = chunks[:k]
    chunk_cots = [c["cot_public"] for c in topk]

    # 4) writer  verifier loop
    full_answer = ""
    verifier_cot = None
    writer_cot = None

    verif = {}
    for attempt in range(1, 6):
        # writer streams, but we only capture the THOUGHT section
        tokens = list(stream_answer(context_prefix +  user_input, plan, topk))
        full = "".join(tokens)
        # extract writer cot between <<THOUGHT>> and <<END_COT>>
        start = full.find("<<THOUGHT>>")
        end   = full.find("<<END_COT>>", start) + len("<<END_COT>>")
        writer_cot = full[start:end].strip()
        # verify
        verif = verify_solution(user_input, plan, full)
        verifier_cot = verif["reason"]
        if verif["verdict"] == "Yes":
            logger.info("Solution verified ✓")
            break
        else:
            logger.warning("Verifier never approved solution; displaying best-effort.")


    # 5) record assistant
    add_to_memory("assistant", full)

    # 6) emit ONE raw JSON blob
    output = {
        "planner": plan,           # full planner dict
        "chunks": topk,            # list of chunk dicts used
        "verifier": verif,         # full verifier JSON
        "writer": full             # full streaming text from writer
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))

    # ── Save into hot-path cache (15 min TTL) ──
    sections = _writer_sections(full)          # uses the new helper
    main_display = (
        "THOUGHT:\n"  + sections["THOUGHT"]  + "\n\n"
        "ACTION:\n"   + sections["ACTION"]   + "\n\n"
        "EVIDENCE:\n" + sections["EVIDENCE"]
    )
    set_cached(user_input, {"display": main_display})
    
    # ——— finally return the pipeline result ———
    return {
        "type":     "pipeline",
        "planner":  plan,    # full planner dict
        "chunks":   topk,    # list of chunk dicts used
        "verifier": verif,   # full verifier JSON
        "writer":   full     # full streaming text from writer
     }

# ── tiny helpers for the UI ──────────────────────────────────────────
def clear_hot_cache() -> str:
    """Flush entire Redis + in‑proc cache."""
    from stores_mem_and_cache.cache import _redis, _local_cache
    try:
        _redis.flushdb()
    except Exception:
        pass
    _local_cache.clear()
    return "✅ Response‑cache cleared."

def clear_mem() -> str:
    """Flush STM & LTM."""
    from stores_mem_and_cache.memory import _stm, _redis, LTM_KEY_HASH, LTM_KEY_SET
    _stm.clear()
    try:
        _redis.delete(LTM_KEY_HASH, LTM_KEY_SET)
    except Exception:
        pass
    return "🧹 Chat‑memory cleared."



if __name__ == "__main__":
    while True:
        q = input("\nYou: ")
        if q.lower() in ("exit","quit"):
            break
        result = run_chat_turn(q)
        if not isinstance(result, dict):
           # fallback
           continue
        # pipeline result vs others
        if result["type"] == "pipeline":
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
           # memory, error, off_topic, etc.
           print("\nAssistant:\n", result.get("message", ""))
