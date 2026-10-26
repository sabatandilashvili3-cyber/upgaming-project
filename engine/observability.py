"""Langfuse tracing, optional: with no LANGFUSE_* keys every helper is a no-op.

One investigation = one trace:

    investigation                         (agent)      question in, report summary out
    ├─ claude step 1                      (generation) tokens, cache hits, cost, stop reason
    ├─ search / read_document / ...       (tool)       input, evidence returned, error level
    ├─ claude step 2 ...
    └─ verify                             (evaluator)  sentences kept / dropped, contradictions
    scores: status, findings_kept, sentences_dropped, contradictions, cost_usd

Environment (LANGFUSE_TRACING_ENVIRONMENT) separates live traffic ("production")
from evaluation runs ("eval"), so dashboards never mix the two.
"""
import os
from contextlib import contextmanager, nullcontext

_client = None
_checked = False


def client():
    """The Langfuse client, or None when tracing is not configured."""
    global _client, _checked
    if not _checked:
        _checked = True
        if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
            from langfuse import Langfuse
            _client = Langfuse()          # reads LANGFUSE_PUBLIC_KEY / SECRET_KEY / BASE_URL from the environment
    return _client


def observation(name: str, as_type: str = "span", **kw):
    lf = client()
    return lf.start_as_current_observation(name=name, as_type=as_type, **kw) if lf else nullcontext(None)


@contextmanager
def investigation_trace(question: str, tags=(), metadata=None):
    """Root of one investigation. Yields the root observation (or None)."""
    lf = client()
    if lf is None:
        yield None
        return
    from langfuse import propagate_attributes
    with lf.start_as_current_observation(name="investigation", as_type="agent", input={"question": question}) as root:
        with propagate_attributes(trace_name="investigation", tags=list(tags), metadata=metadata or {}):
            yield root


def flush():
    if client():
        client().flush()


def trace_url(trace_id: str | None):
    lf = client()
    return lf.get_trace_url(trace_id=trace_id) if lf and trace_id else None
