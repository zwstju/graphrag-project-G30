from __future__ import annotations

from typing import Any, Dict, List, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.hybrid.pipeline import answer_query


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question")
    mode: Literal["vector", "graph", "hybrid"] = "hybrid"


class AskResponse(BaseModel):
    query: str
    mode: str
    answer: str
    vector_results: List[Dict[str, Any]]
    graph_results: Dict[str, Any]
    references: List[Dict[str, str]]


app = FastAPI(
    title="GNN Literature HybridRAG API",
    version="0.1.0",
    description="Backend API for VectorRAG / GraphRAG / HybridRAG comparison.",
)

# Development default: allow the frontend to connect from another local port.
# Before public deployment, replace "*" with the actual frontend origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        result = answer_query(request.query, request.mode)
        print("\n===== BACKEND DEBUG =====")
        print("answer repr:", repr(result["answer"]))
        print("answer length:", len(result["answer"]))
        print("===== END BACKEND DEBUG =====\n")

        return AskResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        # Keep the API response clean while still exposing an actionable message
        # during local development.
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline execution failed: {exc}",
        ) from exc
