from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from stage2.pipeline import Stage2System
from stage2.rerank.feature_extractor import (
    FEATURE_NAMES,
    EvidenceFeatureExtractor,
    evidence_dict,
)


LIKELY_CORE69 = [
    "benchmark/gnn_qa_core69.jsonl",
    "benchmark/gnn_qa_core69_audit_filtered.jsonl",
    "benchmark/core69.jsonl",
    "results/final/core69.jsonl",
    "results/final/core69_eval_set.jsonl",
]

LIKELY_AUDITS = [
    "results/final/benchmark_audit_core80.csv",
    "results/benchmark_audit_core80.csv",
    "results/final/benchmark_audit_core80.jsonl",
    "results/benchmark_audit_core80.jsonl",
]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def qid_of(row: Dict[str, Any]) -> str:
    return str(
        row.get("qid")
        or row.get("id")
        or row.get("question_id")
        or ""
    )


def question_of(row: Dict[str, Any]) -> str:
    return str(row.get("question") or row.get("query") or "")


def gold_papers_of(row: Dict[str, Any]) -> set[str]:
    vals = (
        row.get("gold_papers")
        or row.get("gold_paper_ids")
        or row.get("gold_paper")
        or row.get("paper_ids")
        or []
    )
    if isinstance(vals, str):
        vals = [vals]
    return {str(x) for x in vals if x}


def find_benchmark(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    for s in LIKELY_CORE69:
        p = Path(s)
        if p.exists():
            print(f"[auto] using audited benchmark: {p}")
            return p

    p = Path("benchmark/gnn_qa_core80.jsonl")
    if p.exists():
        print(
            "[auto] Core69 file not found; falling back to Core80. "
            "Audit filtering will be attempted."
        )
        return p

    raise FileNotFoundError(
        "Could not find Core69/Core80 benchmark. "
        "Pass --benchmark explicitly."
    )


def find_audit(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    for s in LIKELY_AUDITS:
        p = Path(s)
        if p.exists():
            print(f"[auto] using audit file: {p}")
            return p
    return None


def supported_qids_from_audit(path: Path | None) -> set[str] | None:
    if path is None:
        return None

    supported = set()
    if path.suffix.lower() == ".jsonl":
        rows = load_jsonl(path)
        for r in rows:
            qid = qid_of(r)
            flag = (
                r.get("llm_evidence_supported")
                if "llm_evidence_supported" in r
                else r.get("supported")
            )
            if bool(flag):
                supported.add(qid)
        return supported

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            qid = qid_of(r)
            raw = (
                r.get("llm_evidence_supported")
                or r.get("supported")
                or r.get("status")
                or ""
            )
            val = str(raw).strip().lower()
            if val in {"1", "true", "yes", "supported", "llm_evidence_supported"}:
                supported.add(qid)
    return supported


def heldout_qids_from_generation(path: Path) -> set[str]:
    if not path.exists():
        print(f"[warn] generation file missing: {path}")
        return set()
    return {
        qid_of(r)
        for r in load_jsonl(path)
        if qid_of(r)
    }


def candidate_pool(system: Stage2System, query: str, k: int) -> List[Any]:
    # Training must use pre-rerank candidates.
    vector = system.vector.search(query, k)
    graph = system.graph.search(query, k)

    out = []
    seen = set()
    for e in list(vector) + list(graph):
        d = evidence_dict(e)
        key = (
            d.get("source_type"),
            d.get("paper_id"),
            d.get("content"),
        )
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out


def evidence_is_positive(e: Any, gold_papers: set[str]) -> bool:
    d = evidence_dict(e)
    papers = set(d.get("paper_ids") or [])
    if d.get("paper_id"):
        papers.add(d["paper_id"])
    return bool(papers & gold_papers)


def build_pairwise_examples(
    system: Stage2System,
    extractor: EvidenceFeatureExtractor,
    rows: Sequence[Dict[str, Any]],
    candidate_k: int,
    max_pairs_per_query: int,
    seed: int,
) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
    rng = random.Random(seed)
    diffs = []
    pair_qids = []

    stats = {
        "queries_seen": 0,
        "queries_with_pairs": 0,
        "positive_evidence": 0,
        "negative_evidence": 0,
        "pairs": 0,
    }

    for i, row in enumerate(rows, 1):
        qid = qid_of(row)
        query = question_of(row)
        gold = gold_papers_of(row)

        if not query or not gold:
            continue

        stats["queries_seen"] += 1
        candidates = candidate_pool(system, query, candidate_k)
        if not candidates:
            continue

        x = extractor.transform(query, candidates)
        positives = [
            idx for idx, e in enumerate(candidates)
            if evidence_is_positive(e, gold)
        ]
        negatives = [
            idx for idx, e in enumerate(candidates)
            if not evidence_is_positive(e, gold)
        ]

        stats["positive_evidence"] += len(positives)
        stats["negative_evidence"] += len(negatives)

        pairs = [(p, n) for p in positives for n in negatives]
        if not pairs:
            continue

        rng.shuffle(pairs)
        pairs = pairs[:max_pairs_per_query]
        stats["queries_with_pairs"] += 1

        for p, n in pairs:
            diffs.append(x[p] - x[n])
            pair_qids.append(qid)

        stats["pairs"] += len(pairs)

        if i % 5 == 0:
            print(
                f"[pairs] {i}/{len(rows)} queries, "
                f"current pairs={len(diffs)}"
            )

    if not diffs:
        raise RuntimeError(
            "No training pairs were built. Check gold_papers and candidate evidence provenance."
        )

    return np.asarray(diffs, dtype=np.float32), pair_qids, stats


def sigmoid_neg(z: np.ndarray) -> np.ndarray:
    # sigmoid(-z), numerically stable
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = np.exp(-z[pos]) / (1.0 + np.exp(-z[pos]))
    out[~pos] = 1.0 / (1.0 + np.exp(z[~pos]))
    return out


def pairwise_accuracy(x: np.ndarray, w: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0
    return float(np.mean((x @ w) > 0.0))


def train_projected_logistic(
    x_train: np.ndarray,
    x_val: np.ndarray,
    lr: float,
    l2: float,
    epochs: int,
    patience: int,
) -> tuple[np.ndarray, Dict[str, Any]]:
    # Nonnegative interpretable initialization.
    w = np.full((x_train.shape[1],), 1.0 / x_train.shape[1], dtype=np.float64)

    best_w = w.copy()
    best_val = -1.0
    best_epoch = 0
    stale = 0
    history = []

    for epoch in range(1, epochs + 1):
        z = x_train @ w
        prob_bad = sigmoid_neg(z)
        grad = -(x_train.T @ prob_bad) / len(x_train) + l2 * w

        w -= lr * grad
        # Keep signal weights nonnegative for interpretability.
        w = np.maximum(w, 0.0)

        if w.sum() <= 1e-12:
            w[:] = 1.0 / len(w)
        else:
            w /= w.sum()

        train_acc = pairwise_accuracy(x_train, w)
        val_acc = pairwise_accuracy(x_val, w) if len(x_val) else train_acc
        history.append((epoch, train_acc, val_acc))

        if val_acc > best_val + 1e-6:
            best_val = val_acc
            best_w = w.copy()
            best_epoch = epoch
            stale = 0
        else:
            stale += 1

        if epoch == 1 or epoch % 25 == 0:
            print(
                f"[train] epoch={epoch:4d} "
                f"train_pair_acc={train_acc:.4f} "
                f"val_pair_acc={val_acc:.4f} "
                f"weights={np.round(w, 4).tolist()}"
            )

        if stale >= patience:
            print(f"[train] early stopping at epoch {epoch}")
            break

    return best_w.astype(np.float32), {
        "best_epoch": best_epoch,
        "best_val_pair_accuracy": best_val,
        "final_train_pair_accuracy": pairwise_accuracy(x_train, best_w),
        "history_tail": history[-10:],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.json")
    ap.add_argument("--benchmark", default=None)
    ap.add_argument("--audit", default=None)
    ap.add_argument(
        "--generation-results",
        default="results/final/generation_outputs.jsonl",
        help="Core32 results; qids here are held out from training.",
    )
    ap.add_argument("--output", default="models/evidence_reranker_ltr.json")
    ap.add_argument("--candidate-k", type=int, default=12)
    ap.add_argument("--max-pairs-per-query", type=int, default=40)
    ap.add_argument("--val-ratio", type=float, default=0.20)
    ap.add_argument("--lr", type=float, default=0.12)
    ap.add_argument("--l2", type=float, default=0.02)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--patience", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    benchmark_path = find_benchmark(args.benchmark)
    rows = load_jsonl(benchmark_path)

    audit_path = find_audit(args.audit)
    supported_qids = supported_qids_from_audit(audit_path)
    if supported_qids is not None:
        rows = [r for r in rows if qid_of(r) in supported_qids]
        print(f"[audit] kept {len(rows)} supported benchmark rows")
    elif "69" not in benchmark_path.name.lower():
        print(
            "[warn] No audit filter found and benchmark does not look like Core69. "
            "Training will continue, but for the final experiment use only audited rows."
        )

    heldout = heldout_qids_from_generation(Path(args.generation_results))
    before = len(rows)
    rows = [r for r in rows if qid_of(r) not in heldout]
    print(
        f"[holdout] excluded {before - len(rows)} Core32 rows; "
        f"{len(rows)} rows remain for LTR training/validation"
    )

    if len(rows) < 8:
        raise RuntimeError(
            "Too few non-held-out rows. Check benchmark/audit/generation paths."
        )

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_val = max(1, round(len(rows) * args.val_ratio))
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    print(
        f"[split] train queries={len(train_rows)}, "
        f"validation queries={len(val_rows)}"
    )

    system = Stage2System(args.config)
    extractor = EvidenceFeatureExtractor(system.vector.model)

    print("[build] training pairs")
    x_train, train_pair_qids, train_stats = build_pairwise_examples(
        system,
        extractor,
        train_rows,
        args.candidate_k,
        args.max_pairs_per_query,
        args.seed,
    )

    print("[build] validation pairs")
    x_val, val_pair_qids, val_stats = build_pairwise_examples(
        system,
        extractor,
        val_rows,
        args.candidate_k,
        args.max_pairs_per_query,
        args.seed + 1,
    )

    print(
        f"[matrix] train={x_train.shape}, val={x_val.shape}"
    )

    w, metrics = train_projected_logistic(
        x_train=x_train,
        x_val=x_val,
        lr=args.lr,
        l2=args.l2,
        epochs=args.epochs,
        patience=args.patience,
    )

    payload = {
        "model_type": "pairwise_projected_logistic_ltr",
        "label_type": "weak_supervision_gold_paper_provenance",
        "feature_names": FEATURE_NAMES,
        "weights": [float(x) for x in w],
        "training": {
            "benchmark": str(benchmark_path),
            "audit": str(audit_path) if audit_path else None,
            "heldout_generation_results": args.generation_results,
            "heldout_qids": sorted(heldout),
            "train_query_count": len(train_rows),
            "validation_query_count": len(val_rows),
            "candidate_k": args.candidate_k,
            "max_pairs_per_query": args.max_pairs_per_query,
            "seed": args.seed,
            "train_pair_stats": train_stats,
            "validation_pair_stats": val_stats,
        },
        "metrics": metrics,
        "note": (
            "Weights are learned from weak paper-level provenance supervision. "
            "Core32 generation qids are excluded from training."
        ),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== learned weights ===")
    for name, val in zip(FEATURE_NAMES, w):
        print(f"{name:>10s}: {val:.6f}")
    print(
        f"\ntrain pair acc: {metrics['final_train_pair_accuracy']:.4f}"
    )
    print(
        f"val pair acc:   {metrics['best_val_pair_accuracy']:.4f}"
    )
    print(f"saved model: {out}")


if __name__ == "__main__":
    main()
