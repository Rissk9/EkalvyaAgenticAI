"""
FastAPI application — entry point.
Run with: uvicorn backend.main:app --reload
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.config import get_settings
from backend.graph import get_graph


# ---------------------------------------------------------------------------
# Warm up on startup (load PDF, build vector store, compile graph)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting up — loading resources...")
    get_graph()            # triggers get_vectorstore → get_embeddings internally
    print("✅ Ready!")
    yield


app = FastAPI(
    title="Pathfinder — AI Career Mentor API",
    description="LangGraph-powered career mentor for Indian tech students",
    version="1.0.0",
    lifespan=lifespan,
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    github_username: str | None = None
    leetcode_username: str | None = None
    summary: str = ""           # pass back the previous summary for multi-turn


class ChatResponse(BaseModel):
    output: str
    summary: str                # updated summary to return to client for next turn


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    graph = get_graph()

    state_input = {
        "input": req.message,
        "summary": req.summary,
        "data": {},
        "decision": {},
        "output": "",
    }
    if req.github_username:
        state_input["github_username"] = req.github_username
    if req.leetcode_username:
        state_input["leetcode_username"] = req.leetcode_username

    result = graph.invoke(state_input)

    return ChatResponse(
        output=result["output"],
        summary=result.get("summary", req.summary),
    )
