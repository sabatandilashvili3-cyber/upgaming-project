"""A scripted LLM for tests: plays back a fixed list of turns.

Lets the whole agent loop (tools, ledger, verification, abstention, error paths) be
tested deterministically, with no API key and no cost. Each script entry is either a
Turn to return or an Exception to raise.
"""
from .base import LLM, Conversation, ToolResult, Turn, Usage


class FakeConversation(Conversation):
    def __init__(self, script, log):
        self.script, self.log = list(script), log

    def step(self) -> Turn:
        if not self.script:
            return Turn("", [], "end_turn")
        nxt = self.script.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        self.log.append(("turn", nxt))
        return nxt

    def add_tool_results(self, results: list[ToolResult]):
        self.log.append(("results", results))

    def add_user_text(self, text: str):
        self.log.append(("user", text))


class FakeLLM(LLM):
    name, model = "fake", "fake-1"

    def __init__(self, script):
        self.script, self.log = script, []

    def start(self, system, user_text, tools) -> Conversation:
        self.log.append(("start", system, user_text, [t.name for t in tools]))
        return FakeConversation(self.script, self.log)

    def cost_usd(self, usage: Usage) -> float:
        return 0.0
