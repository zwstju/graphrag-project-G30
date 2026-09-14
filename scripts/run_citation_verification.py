"""Verify claims in the exact saved Hybrid+Rerank generation outputs."""

import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from scripts.run_generation_evaluation import read_jsonl, retry_call
from stage2.llm.client import LLMClient
from stage2.schema import Evidence
from stage2.verify.citation import verify_claims_llm


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", default="results/final/generation_outputs.jsonl")
    parser.add_argument("--output-dir", default="results/final")
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    started = datetime.now(timezone.utc)
    source_path = Path(args.generation)
    source = [
        record for record in read_jsonl(source_path)
        if record["method"] == "structured_hybrid_rerank" and record.get("answer")
    ]
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    client = LLMClient()
    rows = []
    detail_path = output_dir / "citation_outputs.jsonl"
    with detail_path.open("w", encoding="utf-8") as detail_handle:
        for index, record in enumerate(source, start=1):
            print(f"[{index}/{len(source)}] verify exact answer {record['qid']}", flush=True)
            original_answer = record["answer"]
            original_hash = text_sha256(original_answer)
            evidence = [Evidence(**item) for item in record["evidence"]]
            verification_start = time.perf_counter()
            result, attempts, errors = retry_call(
                lambda: verify_claims_llm(record["question"], original_answer, evidence, client),
                args.max_attempts,
            )
            latency = time.perf_counter() - verification_start
            if text_sha256(record["answer"]) != original_hash:
                raise RuntimeError(f"Saved answer changed during verification: {record['qid']}")
            claims = result.get("claims", []) if result else []
            supported = sum(bool(claim.get("supported")) for claim in claims)
            total = len(claims)
            row = {
                "qid": record["qid"], "qtype": record["qtype"],
                "method": "structured_hybrid_rerank_posthoc_verification",
                "answer_sha256": original_hash, "total_claims": total if result else "",
                "supported_claims": supported if result else "",
                "unsupported_claims": total - supported if result else "",
                "claim_support_rate": supported / total if result and total else (0.0 if result else ""),
                "citation_validity": record["citation_check"]["citation_validity"],
                "verification_latency_s": latency, "verification_attempts": attempts,
                "verification_error": " | ".join(errors) if result is None else "",
            }
            rows.append(row)
            detail = dict(row)
            detail.update({
                "question": record["question"], "answer": original_answer,
                "answer_source": args.generation, "answer_regenerated": False,
                "evidence": record["evidence"], "verification": result,
                "attempt_errors": errors,
            })
            detail_handle.write(json.dumps(detail, ensure_ascii=False) + "\n")
            detail_handle.flush(); write_csv(output_dir / "citation_eval.csv", rows)
    completed = datetime.now(timezone.utc)
    successful = [row for row in rows if row["verification_error"] == ""]
    total_claims = sum(int(row["total_claims"]) for row in successful)
    supported_claims = sum(int(row["supported_claims"]) for row in successful)
    summary = {
        "source_generation_artifact": args.generation,
        "answer_regenerated": False,
        "answers": len(rows), "successfully_verified_answers": len(successful),
        "verification_failures": len(rows) - len(successful),
        "total_claims": total_claims, "supported_claims": supported_claims,
        "unsupported_claims": total_claims - supported_claims,
        "citation_support_rate": supported_claims / total_claims if total_claims else None,
        "mean_citation_validity": sum(float(row["citation_validity"]) for row in rows) / len(rows) if rows else None,
        "total_verification_latency_s": sum(row["verification_latency_s"] for row in rows),
        "mean_verification_latency_s": sum(row["verification_latency_s"] for row in rows) / len(rows) if rows else None,
        "total_api_attempts": sum(row["verification_attempts"] for row in rows),
        "retry_count": sum(max(0, row["verification_attempts"] - 1) for row in rows),
        "verification_max_tokens": client.effective_max_tokens(1200),
        "llm_model": client.model, "llm_api_base": client.base,
        "started_at_utc": started.isoformat(), "completed_at_utc": completed.isoformat(),
        "runtime_seconds": (completed - started).total_seconds(),
    }
    (output_dir / "citation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
