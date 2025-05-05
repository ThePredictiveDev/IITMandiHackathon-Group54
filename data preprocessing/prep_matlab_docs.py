#!/usr/bin/env python3
import argparse, csv, hashlib, html, json, re, time, sys
from collections import defaultdict
from urllib.parse import urlparse
from langdetect import detect
from tqdm import tqdm

# ─────────────── Patterns ────────────────
ERROR_RE    = re.compile(r"MATLAB:[A-Za-z0-9:_-]+")
FUNC_RE     = re.compile(r"\b[A-Z][A-Za-z0-9_]{2,}\(")
RELEASE_RE  = re.compile(r"R20\d\d[ab]", flags=re.I)
TOKEN_RE    = re.compile(r"\w+|[^\s\w]", re.UNICODE)
URL_PAT     = re.compile(r"https?://|www\.")
CRUMB_PAT   = re.compile(r">")                   # breadcrumbs
SHORT_WORDS = re.compile(r"^(Home|Back|Next)$", re.I)

# ────────────── Helpers ────────────────
def md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", "ignore")).hexdigest()

def normalise_ws(txt: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(txt)).strip()

def token_count(text: str) -> int:
    return len(TOKEN_RE.findall(text))

def extract_tags(text: str) -> list[str]:
    tags = set(ERROR_RE.findall(text))
    tags.update(m.upper() for m in RELEASE_RE.findall(text))
    tags.update(func[:-1] for func in FUNC_RE.findall(text)[:20])
    return sorted(tags)

def canonical_url(u: str) -> str:
    p = urlparse(u)
    return f"{p.scheme}://{p.netloc}{p.path}"

def eligible(row: dict, url_cols: list[str]) -> bool:
    for c in url_cols:
        v = (row.get(c) or "").lower()
        if "matlab" in v or "mathworks" in v:
            return True
    return False

def is_boilerplate(line: str) -> bool:
    l = line.strip()
    if not l:
        return True
    if URL_PAT.search(l):
        return True
    if CRUMB_PAT.search(l) and len(l.split()) < 10:
        return True
    if SHORT_WORDS.match(l):
        return True
    return False

# ────────────── Main ────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",       required=True)
    ap.add_argument("--out",       required=True)
    ap.add_argument("--min-tokens", type=int, default=10)
    args = ap.parse_args()

    # Detect columns
    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
    url_cols       = [h for h in headers if h.lower().startswith("link-href")]
    if not url_cols:
        sys.exit("ERROR: No Link-href column found.")
    url_col        = url_cols[0]
    link_text_cols = [h for h in headers if h.lower().startswith("link") and not h.lower().endswith("href")]
    text_cols      = [h for h in headers if h.lower().startswith("text")]

    # Aggregate raw segments per URL
    pages = defaultdict(list)
    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not eligible(row, url_cols):
                continue
            url = canonical_url(row.get(url_col) or "")
            # Collect link texts + Text-* fields
            for col in link_text_cols + text_cols:
                seg = normalise_ws(row.get(col) or "")
                if seg:
                    pages[url].append(seg)

    # Build cleaned records
    records = []
    for url, segs in tqdm(pages.items(), desc="Pages"):
        # 1) Split each segment into lines, filter boilerplate
        lines = []
        for seg in segs:
            for line in seg.splitlines():
                l = normalise_ws(line)
                if not is_boilerplate(l):
                    lines.append(l)
        if not lines:
            continue

        # 2) Deduplicate lines (preserve order)
        seen = set(); deduped = []
        for l in lines:
            key = l.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(l)

        # 3) Join deduped lines into full text
        text = " ".join(deduped)

        # 4) Language filter (keep only English)
        try:
            if detect(text[:400]) != "en":
                continue
        except:
            pass

        # 5) Token-count filter
        tokens = token_count(text)
        if tokens < args.min_tokens:
            continue

        # 6) Metadata extraction
        rec = {
            "url":       url,
            "title":     deduped[0][:120] if deduped else "Untitled",
            "markdown":  text,
            "tags":      extract_tags(text),
            "tokens":    tokens,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hash":      md5(text),
        }
        records.append(rec)

    # Deduplicate pages by content hash
    unique = {r["hash"]: r for r in records}.values()

    # Write JSONL
    with open(args.out, "w", encoding="utf-8") as f:
        for r in unique:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(unique)} cleaned pages to {args.out}")

if __name__ == "__main__":
    main()
