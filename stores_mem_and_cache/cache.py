#!/usr/bin/env python3
"""
cache.py — Simple Redis + in‐process fallback cache for entire query responses
"""

import os
import json
import hashlib
import redis

# 1) Redis client (localhost, no auth)
_redis = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

# 2) Fallback in‐process cache
_local_cache = {}

# 3) Generate a key from the query
def _make_key(query: str) -> str:
    norm = query.lower().strip()
    h = hashlib.sha256(norm.encode("utf-8")).hexdigest()
    return f"dlhack:response:{h}"

def get_cached(query: str) -> dict | None:
    key = _make_key(query)
    print(f"[cache] looking up key {key}")
    try:
        payload = _redis.get(key)
        if payload:
            print(f"[cache] hit in Redis for {key}")
            return json.loads(payload)
    except redis.RedisError as e:
        print(f"[cache] Redis error ({e}); falling back to local cache")
        pass
    return _local_cache.get(key)

def set_cached(query: str, data: dict, ttl: int = 15 * 60) -> None:
    key = _make_key(query)
    raw = json.dumps(data, ensure_ascii=False)
    print(f"[cache] setting key {key} with TTL={ttl}s")
    try:
        _redis.set(key, raw, ex=ttl)
        print(f"[cache] stored in Redis for {key}")
    except redis.RedisError:
        print(f"[cache] Redis unavailable, storing locally for {key}")
        _local_cache[key] = data
