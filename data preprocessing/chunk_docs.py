#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path
import re
from tqdm import tqdm

# Attempt HuggingFace tokenizer, else fall back to tiktoken
try:
    from transformers import AutoTokenizer
    use_hf = True
except ImportError:
    use_hf = False
    try:
        import tiktoken
    except ImportError:
        raise ImportError("Please install either 'transformers' or 'tiktoken' to run this script")

# Regex-style tokenizer for HuggingFace fallback
TOKEN_RE = re.compile(r"\w+|[^\s\w]", re.UNICODE)

def chunk_with_hf(text, tokenizer, chunk_size, stride):
    enc = tokenizer(
        text,
        return_overflowing_tokens=True,
        max_length=chunk_size,
        stride=stride,
        truncation=True
    )
    return [
        tokenizer.decode(ids, skip_special_tokens=True).strip()
        for ids in enc["input_ids"]
    ]

def chunk_with_tiktoken(text, enc, chunk_size, stride):
    token_ids = enc.encode(text)
    chunks = []
    for i in range(0, len(token_ids), chunk_size - stride):
        chunk_ids = token_ids[i : i + chunk_size]
        chunks.append(enc.decode(chunk_ids))
    return chunks

def chunk_document(text, tokenizer, chunk_size, stride):
    if use_hf:
        return chunk_with_hf(text, tokenizer, chunk_size, stride)
    else:
        return chunk_with_tiktoken(text, tokenizer, chunk_size, stride)

def count_tokens_hf(tokenizer, text):
    return len(tokenizer.tokenize(text))

def count_tokens_tiktoken(enc, text):
    return len(enc.encode(text))

def main():
    parser = argparse.ArgumentParser(
        description="Split cleaned docs into overlapping token chunks"
    )
    parser.add_argument("--input",  required=True, help="clean_docs.jsonl")
    parser.add_argument("--output", required=True, help="docs_chunks.jsonl")
    parser.add_argument("--model",  default="intfloat/e5-small-v2",
                        help="HF model for tokenizer")
    parser.add_argument("--chunk-size", type=int, default=256,
                        help="tokens per chunk")
    parser.add_argument("--stride",     type=int, default=32,
                        help="overlap between chunks")
    args = parser.parse_args()

    # Initialize tokenizer/encoder
    if use_hf:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    else:
        enc = tiktoken.get_encoding("cl100k_base")

    out_path = Path(args.output)
    with out_path.open("w", encoding="utf-8") as outf, \
         open(args.input, "r", encoding="utf-8") as inf:
        
        for line in tqdm(inf, desc="Chunking documents"):
            doc = json.loads(line)
            text      = doc.get("markdown", "")
            url       = doc.get("url", "")
            title     = doc.get("title", "")
            tags      = doc.get("tags", [])
            doc_hash  = doc.get("hash", "")
            
            # Generate chunks
            chunks = chunk_document(
                text, tokenizer if use_hf else enc,
                args.chunk_size, args.stride
            )
            # Emit each chunk
            for idx, chunk in enumerate(chunks):
                if use_hf:
                    tcount = count_tokens_hf(tokenizer, chunk)
                else:
                    tcount = count_tokens_tiktoken(enc, chunk)

                rec = {
                    "chunk_id":    f"{doc_hash}_{idx}",
                    "source_url":  url,
                    "title":       title,
                    "tags":        tags,
                    "chunk_text":  chunk,
                    "token_count": tcount,
                    "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                outf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✅ Chunking complete, output written to {args.output}")

if __name__ == "__main__":
    main()
