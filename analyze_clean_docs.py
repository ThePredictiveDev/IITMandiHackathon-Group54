import json
from collections import Counter
import argparse

def analyze_clean_docs(jsonl_path):
    docs = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    total = len(docs)
    token_counts = [doc.get('tokens', 0) for doc in docs]
    tags_counts = Counter(tag for doc in docs for tag in doc.get('tags', []))

    # Sample few docs
    sample_docs = docs[:3]

    print(f"Total documents: {total}")
    if total > 0:
        print(f"Tokens per doc: min={min(token_counts)}, median={sorted(token_counts)[len(token_counts)//2]}, max={max(token_counts)}")
        print("Top 10 tags:")
        for tag, count in tags_counts.most_common(10):
            print(f"  {tag}: {count}")
        print("\nSample documents:")
        for i, doc in enumerate(sample_docs, 1):
            print(f"\n--- Document {i} ---")
            print(f"URL: {doc.get('url')}")
            print(f"Title: {doc.get('title')}")
            print(f"Tokens: {doc.get('tokens')}")
            print("Tags:", doc.get('tags'))
            print("Markdown snippet:", doc.get('markdown')[:200], "...")
    else:
        print("No documents found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, help="Path to clean_docs.jsonl")
    args = parser.parse_args()
    analyze_clean_docs(args.jsonl)
