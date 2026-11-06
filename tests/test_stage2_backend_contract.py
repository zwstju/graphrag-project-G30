from src.backend import app_stage2
from stage2 import pipeline
from stage2.schema import Evidence


class FakeStage2System:
    def retrieve(self, query, method):
        return [
            Evidence(
                evidence_id="G1",
                source_type="triple",
                content="VGAE --USES--> GCN",
                paper_id="vgae_2016",
                metadata={"source": "VGAE", "relation": "USES", "target": "GCN"},
            )
        ]

    def answer(self, query, method, verify=False):
        evidence = self.retrieve(query, method)
        return {
            "question": query,
            "method": method,
            "answer": "VGAE 使用 GCN 编码器 [vgae_2016]。",
            "evidence": [item.to_dict() for item in evidence],
            "citation_check": {
                "citation_validity": 1.0,
                "claim_verification": {
                    "claims": [
                        {
                            "claim_id": "C1",
                            "claim": "VGAE 使用 GCN 编码器",
                            "supported": True,
                            "evidence_ids": ["E1"],
                        }
                    ]
                },
            },
            "retrieval_latency_s": 0.1,
            "generation_latency_s": 0.2,
            "total_latency_s": 0.3,
        }


def test_health_and_frontend_do_not_initialize_qa_system(monkeypatch):
    monkeypatch.setattr(app_stage2, "_SYSTEM", None)
    monkeypatch.setattr(
        app_stage2,
        "Stage2System",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("health/frontend must not initialize the QA system")
        ),
    )

    health = app_stage2.health()
    frontend = app_stage2.frontend()

    assert health["status"] == "ok"
    assert health["qa_system_loaded"] is False
    assert str(frontend.path).endswith("frontend_stage2/index.html")


def test_stage2_api_normalizes_pipeline_checks_and_timings(monkeypatch):
    monkeypatch.setattr(app_stage2, "_SYSTEM", FakeStage2System())
    payload = app_stage2.ask(
        app_stage2.AskRequest(
            query="VGAE 使用什么作为编码器？",
            method="full",
            verify=True,
        )
    )

    assert payload["checks"]["citation_validity"] == 1.0
    assert payload["checks"]["claim_verification"]["claims"][0]["supported"] is True
    assert payload["timings"] == {
        "retrieval_latency_s": 0.1,
        "generation_latency_s": 0.2,
        "total_latency_s": 0.3,
    }
    assert payload["evidence"][0]["paper_id"] == "vgae_2016"
    assert payload["graph"]["edges"][0]["relation"] == "USES"


def test_pipeline_exposes_nested_and_backward_compatible_fields(monkeypatch):
    system = pipeline.Stage2System.__new__(pipeline.Stage2System)
    evidence = [Evidence("V1", "vector", "supported text", paper_id="p1")]
    monkeypatch.setattr(system, "retrieve", lambda query, method: evidence)
    monkeypatch.setattr(
        pipeline,
        "answer_question",
        lambda query, selected: "answer [p1]",
    )
    monkeypatch.setattr(
        pipeline,
        "deterministic_citation_check",
        lambda answer, selected: {"citation_validity": 1.0},
    )

    result = system.answer("question", "rerank")

    assert result["checks"] == result["citation_check"]
    assert result["timings"]["retrieval_latency_s"] == result["retrieval_latency_s"]
    assert result["timings"]["generation_latency_s"] == result["generation_latency_s"]
    assert result["timings"]["total_latency_s"] == result["total_latency_s"]
