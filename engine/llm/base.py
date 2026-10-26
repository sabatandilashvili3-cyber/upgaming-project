"""The seam between the agent and any LLM provider.

The agent never sees provider message formats. It works with a Conversation:

    conv = llm.start(system, user_text, tools)
    turn = conv.step()                   # one model call
    conv.add_tool_results([...])         # answer the tool calls of that turn
    turn = conv.step()                   # ... until turn.tool_calls is empty

Each adapter (Claude, a scripted fake for tests) keeps the provider's own message
history inside its Conversation, so provider details never leak into the agent.
"""
from dataclasses import dataclass, field


@dataclass
class ToolSpec:
    name: str
    description: str
    schema: dict                 # JSON Schema for the input; strict: additionalProperties false + required


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class ToolResult:
    call_id: str
    content: str
    is_error: bool = False


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def add(self, other: "Usage"):
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens


@dataclass
class Turn:
    text: str
    tool_calls: list[ToolCall]
    stop: str                    # end_turn | tool_use | max_tokens | refusal | error
    usage: Usage = field(default_factory=Usage)
    detail: str = ""             # refusal category, error message ...


class LLMError(RuntimeError):
    """A failure the caller should show instead of crashing: auth, quota, network, bad request."""

    def __init__(self, kind: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.kind, self.retryable = kind, retryable


class Conversation:
    def step(self) -> Turn:
        raise NotImplementedError

    def add_tool_results(self, results: list[ToolResult]):
        raise NotImplementedError

    def add_user_text(self, text: str):
        raise NotImplementedError


class LLM:
    name: str = "base"
    model: str = ""

    def start(self, system: str, user_text: str, tools: list[ToolSpec]) -> Conversation:
        raise NotImplementedError

    def cost_usd(self, usage: Usage) -> float:
        return 0.0
