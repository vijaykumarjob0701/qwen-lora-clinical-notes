#!/usr/bin/env python3
"""Compare a candidate model against a frozen production baseline.

Promotion rule used by this teaching gate: the candidate must beat the
baseline on every key metric. Swap the dummy scorer for real generations
plus ROUGE / clinical entity overlap in production.

This file never loads patient data. It scores two JSONL files of
{id, prediction, reference} records.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def ngrams(words: list[str], n: int) -> Counter:
    return Counter(tuple(words[i : i + n]) for i in range(max(0, len(words) - n + 1)))


def f1(pred: Iterable[str], ref: Iterable[str]) -> float:
    pc, rc = Counter(pred), Counter(ref)
    overlap = sum((pc & rc).values())
    if overlap == 0:
        return 0.0
    precision = overlap / max(1, sum(pc.values()))
    recall = overlap / max(1, sum(rc.values()))
    return 2 * precision * recall / (precision + recall)


def rouge_l(pred: str, ref: str) -> float:
    """Character-cheap longest common subsequence F1 (teaching stand-in for ROUGE-L)."""
    a, b = tokens(pred), tokens(ref)
    if not a or not b:
        return 0.0
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, start=1):
            nxt = dp[j]
            dp[j] = prev + 1 if x == y else max(dp[j], dp[j - 1])
            prev = nxt
    lcs = dp[-1]
    prec = lcs / len(a)
    rec = lcs / len(b)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def bleuish(pred: str, ref: str) -> float:
    """Unigram + bigram overlap. Good enough to teach a gate; use sacrebleu in prod."""
    p, r = tokens(pred), tokens(ref)
    if not p or not r:
        return 0.0
    uni = f1(p, r)
    bi = f1(ngrams(p, 2), ngrams(r, 2))
    return 0.5 * uni + 0.5 * bi


MED_CUES = {
    "metformin",
    "furosemide",
    "atorvastatin",
    "amlodipine",
    "nitrofurantoin",
    "ceftriaxone",
    "azithromycin",
    "pantoprazole",
    "ondansetron",
    "acetaminophen",
    "ferrous",
}
DX_CUES = {
    "hfref",
    "pneumonia",
    "uti",
    "anemia",
    "diabetes",
    "rhinitis",
    "gastritis",
}


def entity_overlap(pred: str, ref: str) -> float:
    def ents(text: str) -> set[str]:
        toks = set(tokens(text))
        return {e for e in MED_CUES | DX_CUES if e in toks or e in text.lower()}

    return f1(ents(pred), ents(ref))


def load_rows(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows[row["id"]] = row
    return rows


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def score_file(path: Path) -> dict[str, float]:
    rouge, bleu, ents = [], [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            pred, ref = row["prediction"], row["reference"]
            rouge.append(rouge_l(pred, ref))
            bleu.append(bleuish(pred, ref))
            ents.append(entity_overlap(pred, ref))
    return {"rouge_l": mean(rouge), "bleu": mean(bleu), "entity_f1": mean(ents)}


def gate(candidate: dict[str, float], baseline: dict[str, float], keys: list[str]) -> bool:
    return all(candidate[k] > baseline[k] for k in keys)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--report", type=Path, default=Path("mlops/var/eval_report.json"))
    args = p.parse_args()

    keys = ["rouge_l", "bleu", "entity_f1"]
    cand = score_file(args.candidate)
    base = score_file(args.baseline)
    promote = gate(cand, base, keys)
    report = {"candidate": cand, "baseline": base, "promote": promote, "keys": keys}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if promote else 2)


if __name__ == "__main__":
    main()
