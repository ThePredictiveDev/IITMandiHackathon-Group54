#!/usr/bin/env python3
"""
memory.py — In-process STM & Redis-backed LTM for multi-turn chatbot

STM (Short-Term Memory): last N turns in a ring buffer.
LTM (Long-Term Memory): Redis sorted set + hash, capped by age & size.

Usage:
    from memory import add_to_memory, get_memory
"""

import os
import time
import hashlib
import logging
from collections import deque
from typing import List, Dict, Any, Optional
import json
import redis

# ─── Configuration ───────────────────────────────────────────────────────────
STM_MAX_TURNS      = int(os.getenv("STM_MAX_TURNS", 10))
LTM_TTL_SECONDS    = int(os.getenv("LTM_TTL_SECONDS", 400*60))
LTM_MAX_ENTRIES    = int(os.getenv("LTM_MAX_ENTRIES", 1000))
REDIS_HOST         = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT         = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB           = int(os.getenv("REDIS_DB_MEMORY", "1"))
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize in-process STM buffer
_stm: deque[Dict[str, Any]] = deque(maxlen=STM_MAX_TURNS)

# Initialize Redis client for LTM
try:
    _redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
    _redis.ping()
    logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}, DB {REDIS_DB}")
except redis.RedisError as e:
    logger.warning(f"Could not connect to Redis ({e}); LTM disabled")
    _redis = None

# Namespaces
LTM_KEY_SET   = "dlhack:ltm:set"
LTM_KEY_HASH  = "dlhack:ltm:data"


def _now_ts() -> int:
    return int(time.time())


def _hash_content(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def add_to_memory(role: str, content: str) -> None:
    """
    Record a chat turn into STM and conditionally into LTM.
    role: "user" or "assistant"
    content: the message text
    """
    turn = {"role": role, "content": content, "ts": _now_ts()}
    _stm.append(turn)
    logger.debug(f"STM append: {role=} {len(_stm)} turns stored")

    # Promote to LTM?
    if not _redis:
        return

    # Compute key
    h = _hash_content(content)
    score = _now_ts()

    try:
        existing = _redis.zscore(LTM_KEY_SET, h)
        promote = False
        # Promote if first time or older than TTL or STM overflowed
        if existing is None or score - existing >= LTM_TTL_SECONDS or len(_stm) == STM_MAX_TURNS:
            promote = True

        if promote:
            pipeline = _redis.pipeline()
            pipeline.zadd(LTM_KEY_SET, {h: score})
            pipeline.hset(LTM_KEY_HASH, h, content)
            # Evict by age
            cutoff = score - LTM_TTL_SECONDS
            pipeline.zremrangebyscore(LTM_KEY_SET, 0, cutoff)
            # Evict by size
            pipeline.zremrangebyrank(LTM_KEY_SET, 0, -LTM_MAX_ENTRIES - 1)
            pipeline.execute()
            logger.debug(f"LTM promote: {h=} at ts={score}")
    except redis.RedisError as e:
        logger.warning(f"LTM promotion failed: {e}")


def get_memory() -> Dict[str, List[Dict[str, Any]]]:
    """
    Retrieve current STM & top-M LTM entries.
    Returns:
      {
        "stm": List[ {role,content,ts} ],   # up to STM_MAX_TURNS
        "ltm": List[ {role="memory",content,ts} ]  # most recent M ≤ 5
      }
    """
    stm_list = list(_stm)
    ltm_list: List[Dict[str, Any]] = []

    if _redis:
        try:
            # ── Purge any LTM entries older than TTL
            cutoff = time.time() - LTM_TTL_SECONDS
            _redis.zremrangebyscore(LTM_KEY_SET, 0, cutoff)
            # fetch top 5 most recent
            entries = _redis.zrevrange(LTM_KEY_SET, 0, 4, withscores=True)
            for h, ts in entries:
                content = _redis.hget(LTM_KEY_HASH, h)
                if content:
                    ltm_list.append({"role": "memory", "content": content, "ts": int(ts)})
                    # stored content is JSON-encoded string
                    try:
                        data = json.loads(content)
                    except Exception:
                        data = content
                    ltm_list.append({"role": "memory", "content": data, "ts": int(ts)})
        except redis.RedisError as e:
            logger.warning(f"LTM fetch failed: {e}")

    return {"stm": stm_list, "ltm": ltm_list}
