#!/usr/bin/env python3
"""
batch_chatbot_demo.py  – Run a batch of Simulink‑related questions through
the existing chatbot pipeline and save raw outputs.

Requirements:
  • chatbot.py (and its dependencies) must be importable from PYTHONPATH
  • Redis etc. should already be running, just like for normal use
"""

import datetime, json, pathlib
from chatbot_dep import run_chat_turn   # uses the full pipeline you’ve built
import re

# ------------------------------------------------------------------
BATCH_QUESTIONS = [
    # 1‑15 diverse Simulink / MATLAB Real‑Time queries
    "After changing the Receive block’s sample time to 0.1, why is my sine wave still distorted?",
    "Why does my Simulink Real‑Time model show CPU overload warnings?",
    "How do I fix ‘License checkout failed’ when launching Simulink?",
    "What causes buffer overflow in a UDP Receive block on the target?",
    "Simulink Real‑Time: Real‑Time tab disappeared – how do I restore it?",
    "Why are some signals missing from my data logs after streaming?",
    "How can I reduce unsatisfactory real‑time performance on Speedgoat hardware?",
    "I get ‘XYZ:UNKNOWN’ error building my model – what does it mean?",
    "How do I profile task execution time on the real‑time target?",
    "What’s the correct way to increase queue capacity for message bus signals?",
    "Which solver settings are recommended for 1 kHz fixed‑step real‑time execution?",
    "Why can’t I connect to the target computer over Ethernet in MATLAB R2023b?",
    "How do I deploy a Simulink model with CAN I/O blocks to real‑time hardware?",
    "What does the warning ‘parameter not tunable in real‑time’ mean?",
    "How do I install Simulink Real‑Time software updates offline?"
]

OUT_FILE = pathlib.Path("batch_results.txt")

# ------------------------------------------------------------------
def main() -> None:
    """
    Run the batch questions and save ONLY the end–user‑visible part of the
    writer output (THOUGHT / ACTION / EVIDENCE) into batch_results.txt.
    """
    def _compact(writer_txt: str) -> str:
        """Return just THOUGHT / ACTION / EVIDENCE sections."""
        sec = {}
        for tag, end in [("THOUGHT", "COT"), ("ACTION", "ACTION"), ("EVIDENCE", "EVIDENCE")]:
            m = re.search(rf"<<{tag}>>(.*?)<<END_{end}>>", writer_txt, re.S)
            sec[tag] = (m.group(1).strip() if m else "*missing*")
        return (
            "### THOUGHT\n"  + sec["THOUGHT"]  + "\n\n"
            "### ACTION\n"   + sec["ACTION"]   + "\n\n"
            "### EVIDENCE\n" + sec["EVIDENCE"]
        )

    print(f"Running {len(BATCH_QUESTIONS)} questions …")
    with OUT_FILE.open("w", encoding="utf-8") as f:
        timestamp = datetime.datetime.utcnow().isoformat(timespec="seconds")
        f.write(f"# Batch run @ {timestamp} UTC\n\n")

        for idx, q in enumerate(BATCH_QUESTIONS, start=1):
            print(f"[{idx}/{len(BATCH_QUESTIONS)}]  {q}")
            res = run_chat_turn(q)

            if res.get("type") == "pipeline":
                writer_block = _compact(res["writer"])
            else:                                  # off‑topic / errors / memory look‑ups
                writer_block = res.get("message", "(no writer output)")

            f.write(f"## Q{idx}: {q}\n\n{writer_block}\n\n{'-'*80}\n\n")

    print(f"\nDone!  Results written to {OUT_FILE.resolve()}")

# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
