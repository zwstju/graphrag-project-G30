"""Aggregate final measured CSVs and render reproducible figures."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


DISPLAY = {
    "fixed_vector": "Fixed Vector",
    "structured_vector": "Structured Vector",
    "structured_vanilla_hybrid": "Vanilla Hybrid",
    "structured_hybrid_rerank": "Hybrid + Rerank",
}


def read_csv(path: Path) -> list[dict]:
    """读取带 BOM 兼容的 CSV 并返回字典记录。"""
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def number(value):
    """把非空字段转换为浮点数，同时保留缺失值语义。"""
    if value in (None, ""):
        return None
    return float(value)


def mean_present(rows, field):
    """仅对指定字段的非缺失数值计算平均值。"""
    values = [number(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else ""


def write_csv(path: Path, rows: list[dict]):
    """按首条记录的字段顺序写出 UTF-8 CSV。"""
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def generation_aggregate(rows, by_qtype):
    """按方法及可选题型汇总生成成功率、评分和延迟。"""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["qtype"] if by_qtype else "overall")].append(row)
    output = []
    for (method, qtype), members in sorted(groups.items()):
        output.append({
            "method": method, "qtype": qtype, "question_count": len(members),
            "successfully_generated": sum(bool(row.get("answer")) for row in members),
            "successfully_judged": sum(row.get("llm_correctness_relevance_score", "") != "" for row in members),
            "mean_token_f1": mean_present(members, "token_f1"),
            "mean_llm_correctness_relevance_score": mean_present(members, "llm_correctness_relevance_score"),
            "mean_citation_validity": mean_present(members, "citation_validity"),
            "mean_retrieval_latency_s": mean_present(members, "retrieval_latency_s"),
            "mean_generation_latency_s": mean_present(members, "generation_latency_s"),
            "mean_judge_latency_s": mean_present(members, "judge_latency_s"),
            "mean_end_to_end_latency_s": mean_present(members, "end_to_end_latency_s"),
        })
    return output


def grouped_bars(path, labels, series, title, ylabel):
    """根据真实聚合值绘制并保存分组柱状图。"""
    width = 0.8 / max(1, len(series))
    positions = list(range(len(labels)))
    figure, axis = plt.subplots(figsize=(10, 5.5))
    for offset, (name, values) in enumerate(series.items()):
        xs = [position - 0.4 + width / 2 + offset * width for position in positions]
        axis.bar(xs, values, width=width, label=name)
    axis.set_xticks(positions, labels, rotation=15, ha="right")
    axis.set_ylabel(ylabel); axis.set_title(title); axis.legend(); axis.grid(axis="y", alpha=0.25)
    figure.tight_layout(); figure.savefig(path, dpi=180); plt.close(figure)


def main():
    """汇总最终检索、生成、验证和延迟结果并生成图表。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="results/final")
    args = parser.parse_args()
    root = Path(args.dir); figures = root / "figures"; figures.mkdir(parents=True, exist_ok=True)
    retrieval_rows = read_csv(root / "retrieval_ablation.csv")
    retrieval_overall = read_csv(root / "retrieval_overall.csv")
    retrieval_qtype = read_csv(root / "retrieval_by_qtype.csv")
    generation_rows = read_csv(root / "generation_eval.csv")
    generation_overall = generation_aggregate(generation_rows, False)
    generation_qtype = generation_aggregate(generation_rows, True)
    write_csv(root / "generation_overall.csv", generation_overall)
    write_csv(root / "generation_by_qtype.csv", generation_qtype)
    citation = json.loads((root / "citation_summary.json").read_text(encoding="utf-8"))

    retrieval_map = {row["method"]: row for row in retrieval_overall}
    generation_map = {row["method"]: row for row in generation_overall}
    summary = []
    for method in DISPLAY:
        r = retrieval_map[method]; g = generation_map.get(method, {})
        summary.append({
            "method": method, "display_name": DISPLAY[method],
            "retrieval_questions": int(r["question_count"]),
            "paper_retrieval_recall_proxy": r["paper_retrieval_recall_proxy"],
            "paper_evidence_precision_proxy": r["paper_evidence_precision_proxy"],
            "mean_retrieval_latency_s": r["mean_retrieval_latency_s"],
            "generation_questions": int(g["question_count"]) if g else "",
            "mean_token_f1": g.get("mean_token_f1", ""),
            "mean_llm_correctness_relevance_score": g.get("mean_llm_correctness_relevance_score", ""),
            "mean_citation_validity": g.get("mean_citation_validity", ""),
            "mean_generation_latency_s": g.get("mean_generation_latency_s", ""),
            "mean_end_to_end_latency_s": g.get("mean_end_to_end_latency_s", ""),
            "posthoc_claim_support_rate": citation["citation_support_rate"] if method == "structured_hybrid_rerank" else "",
            "mean_verification_latency_s": citation["mean_verification_latency_s"] if method == "structured_hybrid_rerank" else "",
        })
    write_csv(root / "summary_table.csv", summary)

    latency = []
    for row in summary:
        latency.append({
            "method": row["method"], "scope": "overall",
            "retrieval_latency_s": row["mean_retrieval_latency_s"],
            "generation_latency_s": row["mean_generation_latency_s"],
            "end_to_end_latency_s": row["mean_end_to_end_latency_s"],
            "verification_latency_s": row["mean_verification_latency_s"],
        })
    write_csv(root / "latency.csv", latency)

    labels = [row["display_name"] for row in summary]
    grouped_bars(
        figures / "retrieval_metrics_by_method.png", labels,
        {"Recall proxy": [float(row["paper_retrieval_recall_proxy"]) for row in summary],
         "Precision proxy": [float(row["paper_evidence_precision_proxy"]) for row in summary]},
        "Core69 paper-level retrieval proxies", "Mean score",
    )
    qtypes = ["fact", "relation", "comparison", "multi_hop"]
    qtype_map = {(row["method"], row["qtype"]): row for row in retrieval_qtype}
    grouped_bars(
        figures / "results_by_qtype.png", [x.replace("_", " ").title() for x in qtypes],
        {DISPLAY[method]: [float(qtype_map[(method, q)]["paper_retrieval_recall_proxy"]) for q in qtypes]
         for method in DISPLAY},
        "Core69 retrieval recall proxy by qtype", "Mean recall proxy",
    )
    figure, axis = plt.subplots(figsize=(7, 5.5))
    for row in summary:
        axis.scatter(float(row["paper_retrieval_recall_proxy"]), float(row["paper_evidence_precision_proxy"]), s=80)
        axis.annotate(row["display_name"], (float(row["paper_retrieval_recall_proxy"]), float(row["paper_evidence_precision_proxy"])), xytext=(5,5), textcoords="offset points")
    axis.set_xlabel("Paper retrieval recall proxy"); axis.set_ylabel("Paper evidence precision proxy")
    axis.set_title("Core69 precision/recall proxy trade-off"); axis.grid(alpha=0.25)
    figure.tight_layout(); figure.savefig(figures / "precision_recall_proxy.png", dpi=180); plt.close(figure)
    grouped_bars(
        figures / "latency.png", labels,
        {"Retrieval": [float(row["mean_retrieval_latency_s"]) for row in summary],
         "End-to-end": [float(row["mean_end_to_end_latency_s"]) if row["mean_end_to_end_latency_s"] != "" else 0 for row in summary]},
        "Measured mean latency", "Seconds",
    )
    grouped_bars(
        figures / "citation_support.png", ["Hybrid + Rerank"],
        {"Claim support": [float(citation["citation_support_rate"])],
         "Citation validity": [float(citation["mean_citation_validity"])]},
        "Post-hoc citation and claim audit", "Mean rate",
    )
    print(json.dumps({"summary_rows": len(summary), "figures": sorted(path.name for path in figures.glob("*.png"))}, indent=2))


if __name__ == "__main__":
    main()
