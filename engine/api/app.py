"""HTTP API + the web page. Run: uvicorn engine.api.app:app

Endpoints
  GET    /api/health                  database counts, whether an LLM is configured
  GET    /api/documents               uploaded files (one row per file), ?q= filter
  GET    /api/document?doc_id=        one document: text, metadata, version, duplicate links
  POST   /api/documents               upload a file (pdf, docx, html, md, csv, json, jsonl, txt)
  DELETE /api/documents?source=       delete a file and everything derived from it
  POST   /api/search                  retrieval only, with the reason each hit was found
  POST   /api/investigate             server-sent events: each step, then the verified report
  GET    /api/traces, /api/traces/{id}
  GET    /api/eval                    the latest evaluation summaries

Writes (upload, delete) need the ADMIN_TOKEN header when one is configured, so the
public demo can't be emptied by a visitor. Investigations are rate limited per client
and capped by a daily spend budget, because every one costs real money.
"""
import json
import os
import queue
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..agent.investigator import Investigator
from ..ingest.loaders import UnsupportedFormat
from ..llm.base import LLMError
from ..search.retriever import SearchConfig

load_dotenv()
ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "web"
MAX_UPLOAD = 10 * 1024 * 1024
RATE_PER_HOUR = int(os.environ.get("RATE_PER_HOUR", "20"))
DAILY_BUDGET_USD = float(os.environ.get("DAILY_BUDGET_USD", "5"))

app = FastAPI(title="Investigation Engine", version="1.0")
state: dict = {}


def engine():
    if "engine" not in state:
        raise HTTPException(503, "The engine is still starting (loading the embedding model). Try again shortly.")
    return state["engine"]


@app.on_event("startup")
def startup():
    def boot():
        from ..live import Engine
        from ..search.dense import Embedder
        from ..store.db import Store
        store = Store(os.environ["DATABASE_URL"])
        eng = Engine(store, Embedder())
        _ = eng.retriever                                   # build BM25 now, not on the first question
        state["engine"] = eng
        try:
            from ..llm.claude import Claude
            state["llm"] = Claude(effort=os.environ.get("LLM_EFFORT", "high"))
        except LLMError as e:
            state["llm_error"] = str(e)
    threading.Thread(target=boot, daemon=True).start()     # the port opens at once; /api/health reports progress


# ---------------------------------------------------------------- helpers
_hits: dict[str, deque] = defaultdict(deque)


def rate_limit(client: str):
    now, window = time.time(), _hits[client]
    while window and now - window[0] > 3600:
        window.popleft()
    if len(window) >= RATE_PER_HOUR:
        raise HTTPException(429, f"Limit of {RATE_PER_HOUR} investigations per hour reached; try later.")
    window.append(now)


def spent_today(store) -> float:
    with store.pool.connection() as c:
        row = c.execute("select coalesce(sum(cost_usd), 0) as s from traces "
                        "where created_at >= date_trunc('day', now())").fetchone()
    return float(row["s"])


def require_admin(token: str | None):
    expected = os.environ.get("ADMIN_TOKEN")
    if expected and token != expected:
        raise HTTPException(401, "Uploads and deletes need the admin token on this deployment.")


def jsonable(o):
    return json.loads(json.dumps(o, default=str))


# ---------------------------------------------------------------- routes
@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/health")
def health():
    if "engine" not in state:
        return {"status": "starting"}
    s = state["engine"].store.stats()
    return {"status": "ok", **s, "llm": getattr(state.get("llm"), "model", None),
            "llm_error": state.get("llm_error"), "rate_per_hour": RATE_PER_HOUR,
            "daily_budget_usd": DAILY_BUDGET_USD}


@app.get("/api/documents")
def documents(q: str | None = None, limit: int = 100, offset: int = 0):
    return jsonable(engine().store.list_sources(min(limit, 500), offset, q or None))


@app.get("/api/document")
def document(doc_id: str):
    d = engine().store.get_document(doc_id)
    if not d:
        raise HTTPException(404, f"No document {doc_id}")
    return jsonable(d)


@app.post("/api/documents")
async def upload(file: UploadFile = File(...), x_admin_token: str | None = Header(None)):
    require_admin(x_admin_token)
    data = await file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Files up to 10 MB.")
    name = Path(file.filename or "upload").name             # never trust a client path
    try:
        report = engine().ingestor.ingest_file(f"uploads/{name}", data)
    except UnsupportedFormat as e:
        raise HTTPException(415, str(e))
    except Exception as e:                                  # a corrupt PDF etc.: say what failed
        raise HTTPException(422, f"Could not read {name}: {type(e).__name__}: {e}")
    return jsonable(report.__dict__)


@app.delete("/api/documents")
def delete(source: str, x_admin_token: str | None = Header(None)):
    require_admin(x_admin_token)
    n = engine().ingestor.delete_file(source)
    if not n:
        raise HTTPException(404, f"No file {source}")
    return {"deleted_records": n}


class SearchIn(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    mode: str = Field("hybrid", pattern="^(hybrid|bm25|dense)$")
    k: int = Field(10, ge=1, le=50)


@app.post("/api/search")
def search(body: SearchIn):
    cfg = SearchConfig(mode=body.mode, boost=True, per_file=2, k=body.k)
    hits = engine().retriever.search(body.query, cfg)
    return [{"chunk_id": h.chunk_id, "doc_id": h.doc_id, "score": round(h.score, 5), "text": h.text,
             "bm25_rank": h.bm25_rank, "dense_rank": h.dense_rank, "also_in": h.also_in} for h in hits]


class AskIn(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


@app.post("/api/investigate")
def investigate(body: AskIn, request: Request):
    eng = engine()
    if "llm" not in state:
        raise HTTPException(503, state.get("llm_error", "No LLM configured."))
    rate_limit(request.client.host if request.client else "unknown")
    if spent_today(eng.store) >= DAILY_BUDGET_USD:
        raise HTTPException(429, f"Today's LLM budget (${DAILY_BUDGET_USD}) is used up; try tomorrow.")

    events: queue.Queue = queue.Queue()

    def work():
        inv = Investigator(state["llm"], eng.retriever).run(body.question,
                                                            on_event=lambda k, d: events.put((k, d)))
        eng.store.save_trace({"trace_id": inv.trace_id, "question": inv.question, "report": inv.report,
                              "steps": inv.steps, "model": state["llm"].model,
                              "tokens_in": inv.usage.input_tokens + inv.usage.cache_read_tokens
                                           + inv.usage.cache_write_tokens,
                              "tokens_out": inv.usage.output_tokens, "cost_usd": inv.cost_usd,
                              "latency_ms": inv.latency_ms, "error": inv.error})
        events.put(("report", {"trace_id": inv.trace_id, "status": inv.status, "error": inv.error,
                               "report": inv.report, "cost_usd": inv.cost_usd, "latency_ms": inv.latency_ms,
                               "usage": inv.usage.__dict__}))
        events.put(None)

    threading.Thread(target=work, daemon=True).start()

    def stream():
        while (item := events.get()) is not None:
            kind, data = item
            yield f"event: {kind}\ndata: {json.dumps(jsonable(data))}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/traces")
def traces(limit: int = 30):
    return jsonable(engine().store.recent_traces(min(limit, 200)))


@app.get("/api/traces/{trace_id}")
def trace(trace_id: str):
    t = engine().store.get_trace(trace_id)
    if not t:
        raise HTTPException(404, "No such trace")
    return jsonable(t)


@app.get("/api/eval")
def eval_results():
    out = {}
    for p in sorted((ROOT / "eval" / "results").glob("*summary*.json")):
        out[p.stem] = json.loads(p.read_text())
    return {"generated": datetime.now(timezone.utc).isoformat(), "results": out}
