"""Generate and judge answers from the exact saved retrieval evidence."""

import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from stage2.eval.judge import llm_correctness
from stage2.eval.metrics import token_f1
from stage2.llm.client import LLMClient
from stage2.llm.generate import answer_question
from stage2.schema import Evidence
from stage2.verify.citation import deterministic_citation_check


METHODS = ("structured_vector", "structured_vanilla_hybrid", "structured_hybrid_rerank")


def read_jsonl(path: Path) -> list[dict]:
    """读取生成评测所需的 JSONL 记录。"""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def retry_call(function, max_attempts: int = 3):
    """在有界次数内重试空响应或异常，并返回结果、次数和错误列表。"""
    errors = []
    for attempt in range(1, max_attempts + 1):
        try:
            value = function()
            if value is None or (isinstance(value, str) and not value.strip()):
                raise ValueError("empty LLM response")
            return value, attempt, errors
        except Exception as exc:
            errors.append(f"attempt {attempt}: {exc!r}")
    return None, max_attempts, errors


def write_csv(path: Path, rows: list[dict]) -> None:
    """将当前请求级生成结果写入 UTF-8 CSV。"""
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    """使用保存的证据完成三方法生成、统一判分和请求级持久化。"""
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="benchmark/gnn_qa_core32_generation_stratified.jsonl")
    parser.add_argument("--retrieval", default="results/final/retrieval_per_question.jsonl")
    parser.add_argument("--output-dir", default="results/final")
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    started = datetime.now(timezone.utc)
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    items = read_jsonl(Path(args.benchmark))
    retrieval = {record["qid"]: record for record in read_jsonl(Path(args.retrieval))}
    client = LLMClient()
    rows = []
    outputs_path = output_dir / "generation_outputs.jsonl"
    with outputs_path.open("w", encoding="utf-8") as detail_handle:
        for item_index, item in enumerate(items, start=1):
            source = retrieval[item["qid"]]
            for method in METHODS:
                print(f"[{item_index}/{len(items)} {method}] {item['qid']}", flush=True)
                method_data = source["methods"][method]
                evidence = [Evidence(**record) for record in method_data["evidence"]]
                generation_start = time.perf_counter()
                answer, answer_attempts, answer_errors = retry_call(
                    lambda: answer_question(item["question"], evidence, client),
                    args.max_attempts,
                )
                generation_latency = time.perf_counter() - generation_start
                answer = answer or ""
                judge_start = time.perf_counter()
                if answer:
                    judgment, judge_attempts, judge_errors = retry_call(
                        lambda: llm_correctness(
                            item["question"], item["gold_answer"], answer, client
                        ),
                        args.max_attempts,
                    )
                else:
                    judgment, judge_attempts, judge_errors = None, 0, ["generation failed; judge not called"]
                judge_latency = time.perf_counter() - judge_start
                score, reason = judgment if judgment is not None else (None, "")
                citation = deterministic_citation_check(answer, evidence)
                retrieval_latency = float(method_data["retrieval_latency_s"])
                row = {
                    "qid": item["qid"], "qtype": item["qtype"], "difficulty": item["difficulty"],
                    "method": method, "evidence_count": len(evidence),
                    "token_f1": token_f1(answer, item["gold_answer"]) if answer else "",
                    "llm_correctness_relevance_score": score if score is not None else "",
                    "citation_validity": citation["citation_validity"],
                    "retrieval_latency_s": retrieval_latency,
                    "generation_latency_s": generation_latency,
                    "judge_latency_s": judge_latency,
                    "end_to_end_latency_s": retrieval_latency + generation_latency,
                    "answer_attempts": answer_attempts, "judge_attempts": judge_attempts,
                    "generation_error": " | ".join(answer_errors) if not answer else "",
                    "judge_error": " | ".join(judge_errors) if judgment is None else "",
                    "answer": answer, "judge_reason": reason,
                }
                rows.append(row)
                detail = dict(row)
                detail.update({
                    "question": item["question"], "gold_answer": item["gold_answer"],
                    "gold_papers": item["gold_papers"], "evidence": method_data["evidence"],
                    "final_evidence_ids": method_data["final_evidence_ids"],
                    "citation_check": citation,
                    "answer_attempt_errors": answer_errors,
                    "judge_attempt_errors": judge_errors,
                })
                detail_handle.write(json.dumps(detail, ensure_ascii=False) + "\n")
                detail_handle.flush()
                write_csv(output_dir / "generation_eval.csv", rows)
    completed = datetime.now(timezone.utc)
    manifest = {
        "benchmark": args.benchmark, "retrieval_artifact": args.retrieval,
        "methods": list(METHODS), "question_count": len(items), "row_count": len(rows),
        "evidence_count_per_answer": 6, "generation_temperature": 0.0,
        "generation_max_tokens": client.effective_max_tokens(1400), "judge_temperature": 0.0,
        "judge_max_tokens": client.effective_max_tokens(300),
        "llm_model": client.model, "llm_api_base": client.base,
        "started_at_utc": started.isoformat(), "completed_at_utc": completed.isoformat(),
        "runtime_seconds": (completed - started).total_seconds(),
        "generation_failures": sum(not row["answer"] for row in rows),
        "judge_failures": sum(row["llm_correctness_relevance_score"] == "" for row in rows),
        "total_answer_attempts": sum(row["answer_attempts"] for row in rows),
        "total_judge_attempts": sum(row["judge_attempts"] for row in rows),
    }
    (output_dir / "generation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
