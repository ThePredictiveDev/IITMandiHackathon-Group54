#!/usr/bin/env python3
"""
verifier_agent.py — Verifier Agent (CRAG) using Groq client

Uses llama-3.1-8b-instant to verify that the Writer’s solution
is grounded in the Planner’s reasoning. Retries with increasing
leniency up to 5 iterations and at least 60 s of total time.
"""

import time
import json
from groq import Groq

# Instantiate Groq client
API_KEY="" #YOUR_API_KEY_HERE
client = Groq(api_key=API_KEY)

# System prompt for verification
SYSTEM_PROMPT_VERIFY = """
You are a specialized Verifier Agent for MATLAB/Simulink Real-Time troubleshooting. 
Your job is to confirm that the Writer’s proposed solution is both:

  a) Correctly addresses the user’s original query.
  b) Fully grounded in the Planner’s reasoning steps (cot_raw).

You will receive:
• Leniency level: an integer from 1 (strict) to 5 (very lenient).
• Query: the user’s troubleshooting question.
• Planner cot_raw: the full private chain-of-thought used to plan the solution.
• Writer solution: the complete text of the troubleshooting steps generated.

Your task:
1. Determine if ACTION steps are supported by Planner insight.
2. Consider the specified leniency: higher leniency allows broader interpretation.
3. Decide “Yes” only if the Writer’s answer is accurate, complete, and grounded.

Output format:
• Produce **only** valid JSON, nothing else, with fields:
  {
    "verdict": "Yes" or "No",
    "reason": "<your private chain-of-thought explaining the decision>",
    "leniency": <the integer leniency level you applied>
  }
• Do not include any extra commentary, tags, or formatting.
• If you cannot form valid JSON, output EXACTLY INVALID_JSON.
"""


def verify_solution(query: str, plan: dict, solution: str) -> dict:
    """
    Verify the Writer’s solution via LLM, retrying with escalating leniency.

    Args:
      query:    the original user question
      plan:     output from plan_fetch, including plan["cot_raw"]
      solution: the full Writer answer as one string

    Returns:
      A dict with:
        - verdict: "Yes" or "No"
        - reason:  private LLM reasoning
        - leniency: int 1–5
        - iterations: total LLM calls made
        - time_s:    total elapsed seconds
    """
    start = time.time()
    last_result = None

    for iteration in range(1, 1000):  # we’ll break manually
        # compute leniency: 1→2, 2→3, 3→4, 4→5, 5+→5
        leniency = min(5, iteration + 1)

        # build messages
        user_content = (
            f"Leniency level: {leniency}/5\n"
            f"Query: {query}\n"
            f"Planner cot_raw:\n{plan['cot_raw']}\n\n"
            f"Writer solution:\n{solution}\n"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_VERIFY},
            {"role": "user",   "content": user_content},
        ]

        # call Groq
        comp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.0,
            top_p=1.0,
            max_completion_tokens=512,
            stream=False,
            stop=None
        )
        text = comp.choices[0].message.content.strip()

        # handle sentinel
        if text == "INVALID_JSON":
            last_result = {"verdict": "No", "reason": "Invalid JSON from verifier", "leniency": leniency}
        else:
            try:
                parsed = json.loads(text)
                # ensure structure
                assert parsed.get("verdict") in ("Yes", "No")
                assert isinstance(parsed.get("reason"), str)
                assert isinstance(parsed.get("leniency"), int)
                last_result = parsed
            except Exception:
                last_result = {"verdict": "No", "reason": f"Malformed JSON: {text}", "leniency": leniency}

        # if success, break early
        if last_result.get("verdict") == "Yes":
            break

        # check stop conditions: must do ≥5 iterations AND ≥60 s
        elapsed = time.time() - start
        if iteration >= 5 and elapsed >= 60:
            break

    # annotate
    total_time = time.time() - start
    last_result["iterations"] = iteration
    last_result["time_s"]    = round(total_time, 2)
    return last_result
