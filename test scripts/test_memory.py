#!/usr/bin/env python3
"""
test_memory.py — Smoke test for STM & LTM from memory.py
"""

import time
from stores_mem_and_cache.memory import add_to_memory, get_memory, STM_MAX_TURNS, LTM_TTL_SECONDS
import stores_mem_and_cache.memory as memory

def main():
    test_ttl = 10
    print(f"Overriding LTM TTL from {LTM_TTL_SECONDS}s to {test_ttl}s for test\n")
    memory.LTM_TTL_SECONDS = test_ttl
    print(f"STM capacity = {STM_MAX_TURNS}, LTM TTL = {memory.LTM_TTL_SECONDS}s\n")

    # 1) Fill STM beyond its max to force LTM promotion by overflow
    for i in range(STM_MAX_TURNS + 2):
        user_msg = f"User message #{i+1}"
        assistant_msg = f"Assistant response #{i+1}"
        add_to_memory("user", user_msg)
        add_to_memory("assistant", assistant_msg)
        time.sleep(0.1)  # slight delay so timestamps differ

    mem = get_memory()
    print("=== Short-Term Memory (most recent turns) ===")
    for turn in mem["stm"]:
        print(f"[{turn['role']}] {turn['content']} (ts={turn['ts']})")

    print("\n=== Long-Term Memory (promoted entries) ===")
    for entry in mem["ltm"]:
        print(f"- {entry['content']} (ts={entry['ts']})")

    # 2) Now test TTL-based eviction:
    print("\nSleeping for TTL + 1s to test time-based eviction…")
    time.sleep(memory.LTM_TTL_SECONDS + 1)
    mem2 = get_memory()
    print("LTM after TTL expiration:", mem2["ltm"])

if __name__ == "__main__":
    main()
