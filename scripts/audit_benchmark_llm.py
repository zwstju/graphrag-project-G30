"""Gold-aware evidence mining for benchmark validation.

This script uses known gold-paper IDs and the candidate gold answer to find source
passages for audit. Its scores are not retrieval-evaluation results and must never
be reported as such. LLM approval is not human verification.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from stage2.llm.client import LLMClient
from stage2.retrieval.vector import VectorIndex, sha256_file


AUDIT_METHOD = "gold-aware evidence mining for benchmark validation"


def read_jsonl(path: str | Path) -> list[dict]:
    """读取 JSONL 文件，并在遇到无效 JSON 时报告准确行号。"""
    items = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return items


def text_hash(text: str) -> str:
    """计算文本的 SHA-256 摘要，用于证据内容完整性校验。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_retrieval_corpus(chunks: list[dict]) -> dict[str, int]:
    """确认审计语料不含 References，并统计每篇论文的分块数量。"""
    references = [chunk for chunk in chunks if chunk.get("section") == "References"]
    if references:
        raise RuntimeError(
            f"Audit corpus contains {len(references)} canonical References chunks"
        )
    return Counter(chunk.get("paper_id", "") for chunk in chunks)


def select_gold_paper_evidence(
    vector_index: VectorIndex,
    question: str,
    candidate_gold_answer: str,
    gold_papers: list[str],
    top_k_per_paper: int,
) -> tuple[list[dict], dict[str, int]]:
    """按每篇金标准论文的独立配额选择候选答案审计证据。"""
    if top_k_per_paper <= 0:
        raise ValueError("top_k_per_paper must be greater than zero")

    mining_query = f"{question}\nCandidate answer: {candidate_gold_answer}"
    ranked = vector_index.search(mining_query, top_k=vector_index.index.ntotal)
    selected = []
    counts = {}
    evidence_number = 1
    for paper_id in gold_papers:
        paper_evidence = [item for item in ranked if item.paper_id == paper_id]
        paper_evidence = paper_evidence[:top_k_per_paper]
        counts[paper_id] = len(paper_evidence)
        for item in paper_evidence:
            chunk_id = (item.metadata or {}).get("chunk_id", "")
            selected.append(
                {
                    "evidence_id": f"E{evidence_number}",
                    "global_vector_rank": int(item.evidence_id[1:]),
                    "paper_id": item.paper_id,
                    "title": item.title,
                    "section": item.section,
                    "chunk_id": chunk_id,
                    "score": item.retrieval_score,
                    "text_hash": text_hash(item.content),
                    "text": item.content,
                }
            )
            evidence_number += 1
    return selected, counts


def build_balanced_context(
    selected_evidence: list[dict],
    gold_papers: list[str],
    max_chars_per_paper: int,
) -> tuple[str, dict[str, int]]:
    """按论文均衡截断所选证据，并构造供审计模型使用的上下文。"""
    if max_chars_per_paper <= 0:
        raise ValueError("max_chars_per_paper must be greater than zero")

    paper_contexts = []
    used_chars = {}
    for paper_id in gold_papers:
        items = [item for item in selected_evidence if item["paper_id"] == paper_id]
        if not items:
            used_chars[paper_id] = 0
            continue
        per_evidence_budget = max_chars_per_paper // len(items)
        blocks = []
        for item in items:
            prefix = (
                f"[{item['evidence_id']}] paper={item['paper_id']}; "
                f"title={item['title']}; section={item['section']}; "
                f"chunk_id={item['chunk_id']}; score={item['score']:.6f}\n"
            )
            text_budget = max(0, per_evidence_budget - len(prefix) - 1)
            blocks.append(prefix + item["text"][:text_budget])
        paper_context = "\n".join(blocks)[:max_chars_per_paper]
        used_chars[paper_id] = len(paper_context)
        paper_contexts.append(f"=== Gold paper: {paper_id} ===\n{paper_context}")
    return "\n\n".join(paper_contexts), used_chars


def parse_audit_response(raw: str) -> dict:
    """解析并规范化审计模型返回的 JSON 支持判断。"""
    if not raw.strip():
        raise ValueError("Audit model returned an empty response")
    match = re.search(r"\{.*\}", raw, re.S)
    obj = json.loads(match.group(0) if match else raw)
    supported = obj.get("llm_supported", obj.get("supported", False))
    if isinstance(supported, str):
        supported = supported.strip().casefold() in {"true", "1", "yes", "y"}
    score = float(obj.get("llm_support_score", obj.get("score", 0)) or 0)
    if not 0 <= score <= 1:
        raise ValueError(f"LLM support score is outside [0, 1]: {score}")
    return {
        "llm_supported": bool(supported),
        "llm_support_score": score,
        "reason": str(obj.get("reason", "")),
        "suggested_answer": str(
            obj.get("suggested_answer", obj.get("corrected_answer", ""))
        ).strip(),
    }


def audit_with_retries(
    client: LLMClient,
    prompt: str,
    max_attempts: int,
) -> tuple[dict | None, list[str], list[str], list[str], int]:
    """按有界重试策略调用审计模型，并分别记录解析与传输错误。"""
    if max_attempts <= 0:
        raise ValueError("max_attempts must be greater than zero")
    responses = []
    parse_errors = []
    transport_errors = []
    for attempt in range(1, max_attempts + 1):
        retry_instruction = ""
        if attempt > 1:
            retry_instruction = (
                "\nPrevious output was empty or invalid JSON. Return one complete JSON "
                "object only. Keep reason under 120 words and suggested_answer concise."
            )
        try:
            raw = client.chat(
                [{"role": "user", "content": prompt + retry_instruction}],
                temperature=0,
                max_tokens=1200,
            )
        except Exception as exc:
            transport_errors.append(f"attempt {attempt}: {exc!r}")
            continue
        responses.append(raw)
        try:
            return (
                parse_audit_response(raw),
                responses,
                parse_errors,
                transport_errors,
                attempt,
            )
        except (ValueError, json.JSONDecodeError) as exc:
            parse_errors.append(f"attempt {attempt}: {exc!r}")
    return None, responses, parse_errors, transport_errors, max_attempts


def write_summary_csv(records: list[dict], path: Path) -> None:
    """将详细审计记录写成便于人工浏览的紧凑 CSV 摘要。"""
    fields = [
        "id",
        "qtype",
        "gold_paper_count",
        "selected_evidence_count",
        "all_gold_papers_have_quota",
        "llm_status",
        "llm_supported",
        "llm_support_score",
        "manual_status",
        "reason",
        "suggested_answer",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "id": record["id"],
                    "qtype": record["qtype"],
                    "gold_paper_count": len(record["gold_papers"]),
                    "selected_evidence_count": len(record["selected_evidence"]),
                    "all_gold_papers_have_quota": record[
                        "all_gold_papers_have_quota"
                    ],
                    "llm_status": record["llm_status"],
                    "llm_supported": record["llm_supported"],
                    "llm_support_score": record["llm_support_score"],
                    "manual_status": record["manual_status"],
                    "reason": record["reason"],
                    "suggested_answer": record["suggested_answer"],
                }
            )


def main() -> None:
    """执行金论文感知的基准证据审计并保存结果与复现清单。"""
    run_started_perf = time.perf_counter()
    run_timestamp = datetime.now(timezone.utc).isoformat()
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--benchmark", default="benchmark/gnn_qa_core80.jsonl")
    parser.add_argument("--chunks", default="data/stage2/retrieval_chunks.jsonl")
    parser.add_argument("--output", default="results/core80_audit.jsonl")
    parser.add_argument("--summary-csv")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--top-k-per-paper", type=int, default=3)
    parser.add_argument("--max-chars-per-paper", type=int, default=6000)
    parser.add_argument("--max-audit-attempts", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--qtypes", nargs="*", default=[])
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark)
    corpus_path = Path(args.chunks)
    output_path = Path(args.output)
    summary_path = Path(args.summary_csv) if args.summary_csv else output_path.with_suffix(".csv")
    manifest_path = output_path.with_suffix(".manifest.json")

    with open(args.config, encoding="utf-8") as handle:
        config = json.load(handle)
    if corpus_path != Path(config["retrieval_chunks_path"]):
        raise SystemExit(
            "ERROR: --chunks must match configured retrieval_chunks_path; "
            "the audit must not use structure_chunks or References."
        )

    items = read_jsonl(benchmark_path)
    if args.qtypes:
        requested_types = set(args.qtypes)
        items = [item for item in items if item.get("qtype") in requested_types]
    if args.limit:
        items = items[: args.limit]
    chunks = read_jsonl(corpus_path)
    corpus_paper_counts = validate_retrieval_corpus(chunks)

    model_name = os.getenv("EMBED_MODEL", config["embedding_model"])
    client = LLMClient()
    print(f"Audit method: {AUDIT_METHOD}")
    print(f"Benchmark path: {benchmark_path}")
    print(f"Evidence corpus path: {corpus_path}")
    print(f"Output JSONL: {output_path}")
    print(f"Summary CSV: {summary_path}")
    print(f"Audit model: {client.model}")
    print(f"Embedding model: {model_name}")
    print(f"Top-k per gold paper: {args.top_k_per_paper}")
    print(f"Max context chars per gold paper: {args.max_chars_per_paper}")
    print(f"Benchmark items selected: {len(items)}")
    print("Canonical References chunks in audit corpus: 0")

    vector_index = VectorIndex(model_name, device=args.device).load(
        config["faiss_path"],
        config["vector_meta_path"],
        config["vector_manifest_path"],
        str(corpus_path),
    )

    run_manifest = {
        "audit_method": AUDIT_METHOD,
        "started_at_utc": run_timestamp,
        "benchmark_path": str(benchmark_path),
        "benchmark_sha256": sha256_file(benchmark_path),
        "retrieval_corpus_path": str(corpus_path),
        "retrieval_corpus_sha256": sha256_file(corpus_path),
        "retrieval_corpus_chunk_count": len(chunks),
        "canonical_references_count": 0,
        "vector_index_path": config["faiss_path"],
        "vector_index_sha256": sha256_file(config["faiss_path"]),
        "embedding_model": model_name,
        "embedding_device": vector_index.device,
        "audit_model": client.model,
        "audit_api_base": client.base,
        "top_k_per_gold_paper": args.top_k_per_paper,
        "max_context_chars_per_gold_paper": args.max_chars_per_paper,
        "max_audit_attempts": args.max_audit_attempts,
        "selected_item_count": len(items),
        "selected_qids": [item["qid"] for item in items],
        "output_jsonl": str(output_path),
        "summary_csv": str(summary_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(run_manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    records = []
    original_gold_answers = {item["qid"]: item["gold_answer"] for item in items}
    with output_path.open("w", encoding="utf-8") as output:
        for number, item in enumerate(items, start=1):
            qid = item["qid"]
            gold_papers = item.get("gold_papers", [])
            selected, evidence_counts = select_gold_paper_evidence(
                vector_index,
                item["question"],
                item["gold_answer"],
                gold_papers,
                args.top_k_per_paper,
            )
            context, context_chars = build_balanced_context(
                selected, gold_papers, args.max_chars_per_paper
            )
            quota_met = all(
                evidence_counts.get(paper_id, 0) == args.top_k_per_paper
                for paper_id in gold_papers
            )

            print(f"[{number}/{len(items)}] audit {qid}")
            print(f"  gold papers: {', '.join(gold_papers)}")
            for paper_id in gold_papers:
                chunk_ids = [
                    evidence["chunk_id"]
                    for evidence in selected
                    if evidence["paper_id"] == paper_id
                ]
                print(f"  {paper_id}: {', '.join(chunk_ids) or 'NO CHUNKS'}")

            audit_result = {
                "llm_supported": False,
                "llm_support_score": 0.0,
                "reason": "",
                "suggested_answer": "",
            }
            raw_response = ""
            raw_responses = []
            parse_errors = []
            transport_errors = []
            audit_attempts = 0
            audit_error = ""
            if not quota_met:
                missing = [
                    paper_id
                    for paper_id in gold_papers
                    if evidence_counts.get(paper_id, 0) < args.top_k_per_paper
                ]
                audit_result["reason"] = (
                    "Evidence quota was not met for gold papers: " + ", ".join(missing)
                )
                llm_status = "llm_evidence_not_supported"
            else:
                prompt = f'''You are performing a strict benchmark evidence audit.
This is {AUDIT_METHOD}, not retrieval evaluation.
Use only the supplied evidence. Check every key clause, especially for comparison
and multi-hop questions. Evidence from every declared gold paper is included.

Return JSON only:
{{"llm_supported": true, "llm_support_score": 0.0, "reason": "...", "suggested_answer": "..."}}
Keep reason under 120 words. If no correction is needed, suggested_answer may be empty.

Question: {item['question']}
Candidate gold answer: {item['gold_answer']}
Evidence:\n{context}'''
                try:
                    (
                        parsed_result,
                        raw_responses,
                        parse_errors,
                        transport_errors,
                        audit_attempts,
                    ) = audit_with_retries(
                        client, prompt, args.max_audit_attempts
                    )
                    if raw_responses:
                        raw_response = raw_responses[-1]
                    if parsed_result is None:
                        error_parts = parse_errors + transport_errors
                        audit_error = "All audit attempts failed: " + "; ".join(
                            error_parts
                        )
                        audit_result["reason"] = audit_error
                        llm_status = "llm_audit_error"
                    else:
                        audit_result = parsed_result
                        llm_status = (
                            "llm_evidence_supported"
                            if audit_result["llm_supported"]
                            else "llm_evidence_not_supported"
                        )
                except Exception as exc:
                    audit_error = repr(exc)
                    if raw_responses:
                        raw_response = raw_responses[-1]
                    audit_result["reason"] = audit_error
                    llm_status = "llm_audit_error"

            record = {
                "id": qid,
                "question": item["question"],
                "qtype": item.get("qtype", ""),
                "candidate_status": "candidate",
                "candidate_gold_answer": item["gold_answer"],
                "gold_papers": gold_papers,
                "evidence_mining_method": AUDIT_METHOD,
                "evidence_query_uses_candidate_answer": True,
                "top_k_per_gold_paper": args.top_k_per_paper,
                "max_context_chars_per_gold_paper": args.max_chars_per_paper,
                "gold_paper_corpus_chunk_counts": {
                    paper_id: corpus_paper_counts.get(paper_id, 0)
                    for paper_id in gold_papers
                },
                "selected_evidence_counts": evidence_counts,
                "all_gold_papers_have_quota": quota_met,
                "context_chars_by_gold_paper": context_chars,
                "selected_evidence": selected,
                "audit_model": client.model,
                "llm_status": llm_status,
                **audit_result,
                "manual_status": "unreviewed",
                "audit_error": audit_error,
                "raw_audit_response": raw_response,
                "raw_audit_responses": raw_responses,
                "audit_parse_errors": parse_errors,
                "audit_transport_errors": transport_errors,
                "audit_attempts": audit_attempts,
                "audit_run_started_at_utc": run_timestamp,
                "benchmark_sha256": run_manifest["benchmark_sha256"],
                "retrieval_corpus_sha256": run_manifest[
                    "retrieval_corpus_sha256"
                ],
            }
            if item["gold_answer"] != original_gold_answers[qid]:
                raise RuntimeError(f"Automatic audit modified gold_answer for {qid}")
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            records.append(record)

    write_summary_csv(records, summary_path)
    run_manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    run_manifest["runtime_seconds"] = time.perf_counter() - run_started_perf
    run_manifest["completed_item_count"] = len(records)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(run_manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    status_counts = Counter(record["llm_status"] for record in records)
    print(f"Saved canonical audit JSONL: {output_path} ({len(records)} rows)")
    print(f"Saved audit manifest: {manifest_path}")
    print(f"Saved compact summary CSV: {summary_path}")
    print(f"LLM status counts: {dict(sorted(status_counts.items()))}")
    print(f"Audit runtime seconds: {run_manifest['runtime_seconds']:.3f}")
    print("Manual status for every item: unreviewed")
    print("Candidate gold answers modified: 0")


if __name__ == "__main__":
    main()
