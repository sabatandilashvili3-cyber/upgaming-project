"""Claude adapter (official Anthropic SDK).

Choices, each one deliberate:
- claude-opus-5-5 with an explicit effort level (its default is "medium"); the effort
  is configurable so the evaluation can compare levels.
- Prompt caching: the system prompt and tool list never change between steps, and each
  step only appends to the history, so automatic caching re-reads the whole prefix at
  about a tenth of the price. usage.cache_read_input_tokens proves it in the trace.
- Tools are strict (schema-validated inputs). Forced tool choice is not available on
  this model, so the agent is told in the prompt to finish with submit_report.
- Server-side refusal fallback is on: if a safety classifier declines a request, the
  API re-runs it on a fallback model inside the same call.
- The full assistant content (including thinking blocks) is appended back unchanged,
  as the API requires for multi-step tool use.
- Errors: the SDK already retries 408/409/429/5xx and connection errors (max_retries);
  anything that still fails is mapped to an LLMError the agent reports instead of crashing.
"""
import os

import anthropic

from .base import LLM, Conversation, LLMError, ToolCall, ToolResult, ToolSpec, Turn, Usage

# models that accept the server-side refusal fallback ("default" form); Haiku 5.5 has none
FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
# $ per million tokens (input, output, cache read); cache writes cost 1.25x input
PRICES = {"claude-opus-5-5": (4.00, 20.00, 0.20), "claude-sonnet-5-5": (2.00, 10.00, 0.20),
          "claude-haiku-5-5": (0.10, 0.50, 0.01)}


def _usage(u) -> Usage:
    return Usage(input_tokens=u.input_tokens or 0, output_tokens=u.output_tokens or 0,
                 cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
                 cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0)


class ClaudeConversation(Conversation):
    def __init__(self, llm: "Claude", system: str, user_text: str, tools: list[ToolSpec]):
        self.llm = llm
        self.system = system
        self.tools = [{"name": t.name, "description": t.description, "input_schema": t.schema, "strict": True}
                      for t in tools]
        self.messages = [{"role": "user", "content": user_text}]

    def step(self) -> Turn:
        try:
            extra = ({"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
                     if self.llm.model in FALLBACK_MODELS else {})
            resp = self.llm.client.beta.messages.create(
                model=self.llm.model,
                max_tokens=self.llm.max_tokens,
                system=self.system,
                tools=self.tools,
                messages=self.messages,
                output_config={"effort": self.llm.effort},
                cache_control={"type": "ephemeral"},
                **extra,
            )
        except anthropic.AuthenticationError as e:
            raise LLMError("auth", f"Claude rejected the API key: {e.message}") from e
        except anthropic.PermissionDeniedError as e:
            raise LLMError("permission", f"The API key cannot use {self.llm.model}: {e.message}") from e
        except anthropic.NotFoundError as e:
            raise LLMError("not_found", f"Unknown model or endpoint: {e.message}") from e
        except anthropic.RateLimitError as e:
            raise LLMError("rate_limit", "Rate limited after retries; try again shortly.", retryable=True) from e
        except anthropic.BadRequestError as e:
            raise LLMError("bad_request", f"Request rejected: {e.message}") from e
        except anthropic.APIStatusError as e:
            raise LLMError("server", f"Claude API error {e.status_code}: {e.message}",
                           retryable=e.status_code >= 500) from e
        except anthropic.APIConnectionError as e:
            raise LLMError("network", "Could not reach the Claude API.", retryable=True) from e

        self.messages.append({"role": "assistant", "content": resp.content})
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, b.input) for b in resp.content if b.type == "tool_use"]
        detail = ""
        if resp.stop_reason == "refusal" and getattr(resp, "stop_details", None):
            detail = f"{resp.stop_details.category}: {resp.stop_details.explanation}"
        return Turn(text, calls, resp.stop_reason or "end_turn", _usage(resp.usage), detail)

    def add_tool_results(self, results: list[ToolResult]):
        # every result of one turn goes back in ONE user message
        self.messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": r.call_id, "content": r.content, "is_error": r.is_error}
            for r in results]})

    def add_user_text(self, text: str):
        self.messages.append({"role": "user", "content": text})


class Claude(LLM):
    name = "claude"

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "high", max_tokens: int = 16000,
                 max_retries: int = 3, timeout: float = 120.0):
        if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
            raise LLMError("auth", "ANTHROPIC_API_KEY is not set (put it in .env or the host's secrets).")
        self.model, self.effort, self.max_tokens = model, effort, max_tokens
        self.client = anthropic.Anthropic(max_retries=max_retries, timeout=timeout)

    def start(self, system, user_text, tools) -> Conversation:
        return ClaudeConversation(self, system, user_text, tools)

    def cost_usd(self, u: Usage) -> float:
        inp, out, cache_read = PRICES.get(self.model, PRICES["claude-opus-5-5"])
        return (u.input_tokens * inp + u.output_tokens * out + u.cache_read_tokens * cache_read
                + u.cache_write_tokens * inp * 1.25) / 1e6
