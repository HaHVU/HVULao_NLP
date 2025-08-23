import os, glob, json

# === cấu hình ===
INPUT_DIR = r"D:\DatasegLao\HVULao_NLP_clean\Datatrain10k"
OUTPUT_TXT = os.path.join(INPUT_DIR, "sentences_space.txt")

def extract_tokens_from_sentence(sent):
    """
    Hỗ trợ vài biến thể cấu trúc:
    - [ {"word": "...", "label": "..."}, ... ]
    - {"tokens": [ {"word": "..."} , ... ] }
    - {"words": ["...", "...", ...]}
    """
    if isinstance(sent, list):
        toks = []
        for x in sent:
            if isinstance(x, dict):
                w = x.get("word", "")
            else:
                w = str(x)
            w = w.strip()
            if w:
                toks.append(w)
        return toks

    if isinstance(sent, dict):
        if "tokens" in sent and isinstance(sent["tokens"], list):
            return [ (t.get("word","") or "").strip() for t in sent["tokens"] if (t.get("word","") or "").strip() ]
        if "words" in sent and isinstance(sent["words"], list):
            return [ (str(w) or "").strip() for w in sent["words"] if (str(w) or "").strip() ]

    return []  # không nhận diện được

def sentences_from_json(data):
    """
    data có thể là:
    - list các câu
    - dict có key 'sentences' hoặc tương tự
    """
    if isinstance(data, list):
        iterable = data
    elif isinstance(data, dict):
        # thử vài khoá thường gặp
        for k in ("sentences", "data", "items"):
            if k in data and isinstance(data[k], list):
                iterable = data[k]
                break
        else:
            return []
    else:
        return []

    for sent in iterable:
        toks = extract_tokens_from_sentence(sent)
        if toks:
            yield " ".join(toks)

def main():
    json_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.json")))
    if not json_files:
        raise FileNotFoundError(f"No .json files found in: {INPUT_DIR}")

    total = 0
    with open(OUTPUT_TXT, "w", encoding="utf-8", newline="\n") as out:
        for jf in json_files:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            for line in sentences_from_json(data):
                out.write(line + "\n")
                total += 1

    print(f"Done. Wrote {total} sentences to: {OUTPUT_TXT}")

if __name__ == "__main__":
    main()
