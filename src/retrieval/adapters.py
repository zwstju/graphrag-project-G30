from __future__ import annotations

import importlib
import os
from typing import Any, Callable, Dict, List


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_callable(module_name: str, function_name: str) -> Callable[..., Any]:
    module = importlib.import_module(module_name)
    func = getattr(module, function_name, None)
    if not callable(func):
        raise AttributeError(
            f"{module_name}.{function_name} does not exist or is not callable"
        )
    return func


def _mock_vector_results(query: str, top_k: int) -> List[Dict[str, Any]]:
    sample = [
        {
            "paper_id": "vgae_2016",
            "chunk_id": "vgae_2016_mock_001",
            "text": (
                "Variational Graph Auto-Encoders use a graph convolutional "
                "network encoder to parameterize latent node representations."
            ),
            "score": 0.99,
            "title": "Variational Graph Auto-Encoders",
            "section": "Method",
        },
        {
            "paper_id": "gcn_2017",
            "chunk_id": "gcn_2017_mock_001",
            "text": (
                "Graph Convolutional Networks propagate transformed information "
                "over graph neighborhoods for semi-supervised node classification."
            ),
            "score": 0.88,
            "title": "Semi-Supervised Classification with Graph Convolutional Networks",
            "section": "Method",
        },
    ]
    return sample[:top_k]


def _mock_graph_results(query: str, top_k: int) -> Dict[str, Any]:
    triples = [
        {
            "source": "VGAE",
            "source_type": "Model",
            "relation": "USES",
            "target": "GCN",
            "target_type": "Model",
            "paper_id": "vgae_2016",
        },
        {
            "source": "VGAE",
            "source_type": "Model",
            "relation": "SOLVES",
            "target": "Link Prediction",
            "target_type": "Task",
            "paper_id": "vgae_2016",
        },
    ][:top_k]

    return {
        "entities": ["VGAE", "GCN"],
        "triples": triples,
        "paths": [
            {
                "nodes": ["GCN", "VGAE", "Link Prediction"],
                "relations": ["USED_BY", "SOLVES"],
                "score": 0.95,
            }
        ],
        "paper_ids": ["vgae_2016"],
    }


def retrieve_vector(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    module_name = os.getenv("VECTOR_RETRIEVER_MODULE", "src.vector.retriever")
    function_name = os.getenv("VECTOR_RETRIEVER_FUNCTION", "retrieve_vector")
    allow_mock = _env_flag("ALLOW_MOCK_RETRIEVERS", True)

    try:
        func = _load_callable(module_name, function_name)
        result = func(query, top_k=top_k)
        if not isinstance(result, list):
            raise TypeError(
                "Vector retriever must return list[dict] according to interfaces.md"
            )
        return result
    except Exception:
        if not allow_mock:
            raise
        return _mock_vector_results(query, top_k)


def retrieve_graph(query: str, top_k: int = 5) -> Dict[str, Any]:
    module_name = os.getenv("GRAPH_RETRIEVER_MODULE", "src.graph.retriever")
    function_name = os.getenv("GRAPH_RETRIEVER_FUNCTION", "retrieve_graph")
    allow_mock = _env_flag("ALLOW_MOCK_RETRIEVERS", True)

    try:
        func = _load_callable(module_name, function_name)
        result = func(query, top_k=top_k)
        if not isinstance(result, dict):
            raise TypeError(
                "Graph retriever must return dict according to interfaces.md"
            )

        # Normalize missing optional fields so the backend contract is stable.
        return {
            "entities": result.get("entities", []),
            "triples": result.get("triples", []),
            "paths": result.get("paths", []),
            "paper_ids": result.get("paper_ids", []),
        }
    except Exception:
        if not allow_mock:
            raise
        return _mock_graph_results(query, top_k)
