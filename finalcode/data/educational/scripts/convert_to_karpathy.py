#!/usr/bin/env python3
import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


CJK_RE = re.compile(r"[\u4e00-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_'][A-Za-z0-9]+)?")


def tokenize_mixed(text: str) -> list[str]:
    tokens: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if CJK_RE.match(ch):
            tokens.append(ch)
            i += 1
            continue
        m = LATIN_RE.match(text, i)
        if m:
            tokens.append(m.group(0).lower())
            i = m.end()
            continue
        if ch.isdigit():
            tokens.append(ch)
        i += 1
    return tokens


def split_records(records: list[dict], train_ratio: float, val_ratio: float, seed: int) -> dict[str, str]:
    by_subject: dict[str, list[str]] = defaultdict(list)
    for rec in records:
        rel = rec["image_path"]
        subject = rec.get("subject_dir") or rel.split("/")[0]
        by_subject[subject].append(rel)

    rng = random.Random(seed)
    split_by_path: dict[str, str] = {}
    for subject, paths in sorted(by_subject.items()):
        rng.shuffle(paths)
        n = len(paths)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        if n - n_train - n_val < 1 and n >= 3:
            n_train = max(1, n - 2)
            n_val = 1
        for idx, rel in enumerate(paths):
            if idx < n_train:
                split = "train"
            elif idx < n_train + n_val:
                split = "val"
            else:
                split = "test"
            split_by_path[rel] = split
    return split_by_path


def convert(input_jsonl: Path, output_json: Path, seed: int, train_ratio: float, val_ratio: float) -> dict:
    records = [json.loads(line) for line in input_jsonl.open(encoding="utf-8")]
    split_by_path = split_records(records, train_ratio, val_ratio, seed)

    images = []
    sentid = 0
    split_counts = Counter()
    subject_counts = Counter()
    pair_counts = Counter()

    for imgid, rec in enumerate(records):
        rel = rec["image_path"]
        rel_path = Path(rel)
        split = split_by_path[rel]
        split_counts[split] += 1
        subject = rec.get("subject_dir") or rel.split("/")[0]
        subject_counts[subject] += 1

        sentences = []
        sentids = []
        for cap in rec["captions"]:
            raw = cap["zh"].strip()
            sentence = {
                "imgid": imgid,
                "raw": raw,
                "tokens": tokenize_mixed(raw),
                "sentid": sentid,
                "aspect": cap.get("aspect"),
                "raw_en": cap.get("en", "").strip(),
            }
            sentences.append(sentence)
            sentids.append(sentid)
            sentid += 1
            pair_counts[split] += 1

        images.append(
            {
                "filename": rel_path.name,
                "filepath": str(rel_path.parent),
                "imgid": imgid,
                "sentences": sentences,
                "sentids": sentids,
                "split": split,
                "edu_path": rel,
                "subject": subject,
            }
        )

    data = {
        "dataset": "edu_ppt_hq",
        "images": images,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    summary = {
        "input": str(input_jsonl),
        "output": str(output_json),
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": round(1 - train_ratio - val_ratio, 6),
        "images": len(images),
        "sentences": sentid,
        "split_counts": dict(split_counts),
        "pair_counts": dict(pair_counts),
        "subject_counts": dict(subject_counts.most_common()),
        "format": "Karpathy-style COCO/Flickr JSON",
        "caption_language": "zh",
        "extra_sentence_fields": ["aspect", "raw_en"],
        "image_root": "images/",
        "path_rule": "join(image_root, filepath, filename)",
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/home/workspace/fc/edu_ppt_hq_dataset/annotations/images.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/workspace/fc/edu_ppt_hq_dataset/annotations/dataset_edu_ppt_hq_karpathy.json"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    args = parser.parse_args()

    summary = convert(args.input, args.output, args.seed, args.train_ratio, args.val_ratio)
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
