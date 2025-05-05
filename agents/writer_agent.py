#!/usr/bin/env python3
"""
writer_agent.py — Writer Agent (LLM-B) using Groq client

Streams a step-by-step troubleshooting answer with sections:
  THOUGHT: internal reasoning summary (public)
  ACTION: actionable troubleshooting steps
  EVIDENCE: citations with markdown links to documentation

Model: deepseek-r1-distill-llama-70b
"""

import json
from groq import Groq

# Instantiate Groq client
API_KEY="" #YOUR_API_KEY_HERE
client = Groq(api_key=API_KEY)

# System prompt template
SYSTEM_PROMPT = """
You are an expert MATLAB Simulink Real-Time troubleshooting assistant.
Your goal: Generate a **streaming**, step-by-step answer that explains your reasoning,
provides actionable steps, and cites documentation links.

Structure your streamed output exactly as:

<<THOUGHT>> THOUGHT:
Concise, public‐facing summary of why these steps are needed; end with <<END_COT>>

<<ACTION>> ACTION:
1. First step…
2. Second step…
...
n. Final step… <<END_ACTION>>

<<EVIDENCE>> EVIDENCE:
[1](URL) — Explanation of how this source supports ACTION step 1
[2](URL) — Explanation of how this source supports ACTION step 2 <<END_EVIDENCE>>
...

**Rules**:
- After THOUGHT content, end with the literal marker <<END_COT>>.
- Use numbered lists for ACTION.
- Number citations in order of appearance, linking to `source_url`.
- Do not output any extra sections or text beyond THOUGHT, ACTION, EVIDENCE.
- Do NOT output any tags or markers other than exactly:
   <<THOUGHT>>, <<END_COT>>,
   <<ACTION>>, <<END_ACTION>>,
   <<EVIDENCE>>, <<END_EVIDENCE>>.
- Do NOT use `<think>` or any other debugging markers.

**Few‐Shot Example** (for guidance; do not echo in your answer):

Question: Why does my Scope output show a distorted sine wave after changing the Receive block’s Sample time to 0.5?
Plan: There’s a sampling rate mismatch: you set receive to 0.5 while messages come at 0.1, causing skipped data. <<END_COT>>
Chunks:
[1] Matching the send and receive sample times is critical...
[2] When message rates differ, the queue overwrites or skips elements...
Expected streaming format:
<<THOUGHT>> 
THOUGHT:
The receive block’s sample time is slower than the send block’s, causing queued messages to be dropped and the sine wave to appear distorted. 
<<END_COT>>

<<ACTION>> 
ACTION:
1. Set the Receive block’s Sample time to 0.1 to match the Send block.
2. Increase the queue capacity or enable 'Overwrite oldest' to avoid data loss.
3. Stream the data again and confirm the waveform integrity. 
<<END_ACTION>>

<<EVIDENCE>> 
EVIDENCE:
[1](https://in.mathworks.com/help/slrealtime/ug/troubleshoot-overloaded-cpu-from-executing-real-time-application.html) — Details queue drop behavior.
[2](https://in.mathworks.com/help/slrealtime/ug/find-simulink-real-time-support.html) — Describes overwrite/skip behavior. 
<<END_EVIDENCE>>
"""

# ─── Examples for Writer Agent ───
EXAMPLES_WRITER = [
    {
        "query": "Why does my Scope output show a distorted sine wave after changing the Receive block's Sample time to 0.5?",
        "plan_public": "There’s a sampling rate mismatch: you set receive to 0.5 while messages come at 0.1, causing skipped data. <<END_COT>>",
        "chunks": [
            {
                "chunk_id": "a1d9cebbd045df4c56361d2048aedda2_2",
                "chunk_text": "Matching the send and receive sample times is critical. If receive is slower, queued messages will be dropped, distorting the signal.",
                "source_url": "https://in.mathworks.com/help/slrealtime/ug/troubleshoot-overloaded-cpu-from-executing-real-time-application.html"
            },
            {
                "chunk_id": "c2d2edce5362fabc6dcec29317327f07_20",
                "chunk_text": "When message rates differ, the queue overwrites or skips elements if capacity is reached.",
                "source_url": "https://in.mathworks.com/help/slrealtime/ug/find-simulink-real-time-support.html"
            }
        ],
        "response": """<<THOUGHT>> 
        THOUGHT:
The receive block’s sample time is slower than the send block’s, causing queued messages to be dropped and the sine wave to appear distorted. 
<<END_COT>>

<<ACTION>>
ACTION:
1. Set the Receive block’s Sample time to match the Send block (0.1).
2. Verify Queue capacity settings to ensure no data is lost.
3. If intentional backlog analysis is desired, document that signal mismatch will persist. 
<<END_ACTION>>

<<EVIDENCE>> 
EVIDENCE:
[1](https://in.mathworks.com/help/slrealtime/ug/troubleshoot-overloaded-cpu-from-executing-real-time-application.html) — Explains queue drop behavior when sample times mismatch.  
[2](https://in.mathworks.com/help/slrealtime/ug/find-simulink-real-time-support.html) — Describes overwrite and skip behavior in queues. 
<<END_EVIDENCE>>"""
    }
]


def stream_answer(query: str, plan: dict, chunks: list[dict]):
    """
    Streams the troubleshooting answer.

    Args:
      query: original user question
      plan:  planner output dict with cot_public shown to user
      chunks: list of top-k chunk dicts with keys:
        - chunk_id, chunk_text, source_url, match_score, cot_public (optional)

    Yields:
      str: successive text tokens from the model
    """
    # Build messages
    # 1) Core messages
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": plan["cot_public"]},
        {"role": "user",      "content": f"Question: {query}"}
    ]

    # 2) Inject few-shot examples
    for ex in EXAMPLES_WRITER:
        # the “user” side showing the example question + plan + chunks
        messages.append({
            "role": "user",
            "content":
                "Example question:\n"
                f"{ex['query']}\n"
                "Plan PUBLIC:\n"
                f"{ex['plan_public']}\n"
                "Chunks:\n" +
                "\n".join(
                    f"[{i+1}] {c['chunk_text']} (<{c['source_url']}>)"
                    for i, c in enumerate(ex["chunks"])
                )
        })
        # the “assistant” side showing the expected streaming response
        messages.append({
            "role": "assistant",
            "content": ex["response"]
        })

    # 3) Finally, prompt the model with the real context
    messages.append({
        "role": "user",
        "content":
            "Documentation context (first 100 tokens each):\n" +
            "\n".join(
                f"[{i+1}] {c['chunk_text']} (<{c['source_url']}>)"
                for i, c in enumerate(chunks)
            )
    })


    # Request streaming completion
    completion = client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        messages=messages,
        temperature=0.3,
        top_p=1.0,
        max_completion_tokens=2048,
        stream=True,
        stop=["<<END_EVIDENCE>>"]
    )

    # Yield tokens as they arrive
    for chunk in completion:
        delta = chunk.choices[0].delta.content or ""
        yield delta
