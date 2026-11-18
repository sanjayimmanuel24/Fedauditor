"""FedAuditor agent for open models behind an OpenAI-compatible chat API.

Runs the same audit as ``ClaudeAuditor`` (same system prompt, tools, report
schema, fact-checker and scorer) against any server that speaks
``POST /v1/chat/completions`` with tool calling: Ollama, vLLM, llama.cpp
server, LM Studio.

    ollama serve & ollama pull qwen3:8b
    python -m fedauditor.run_audit_grid --agents openai --llm qwen3:8b \
        --base-url http://localhost:11434/v1

(``notebooks/fedauditor_kaggle_qwen.ipynb`` does this on a free Kaggle GPU.)

Small open models need more guard rails than Claude, so the loop:
- parses tool calls written into the text (Hermes / Qwen ``<tool_call>`` form)
  and a report written as plain JSON text instead of a ``submit_report`` call;
- returns invalid JSON arguments to the model as a tool error, and echoes only
  valid JSON back in the history (servers reject a history with bad arguments);
- never sends ``content: null`` (some servers reject it);
- tells the model to submit once the privacy budget is spent, the transcript
  nears the context window, or few turns remain;
- keeps the server's error text when a request fails.

Set ``FEDAUDITOR_TRANSCRIPT_DIR`` (or ``transcript_dir``) to save each audit's
full conversation as ``<scenario id>__<label>__transcript.json`` for debugging.

Uses ``httpx`` only (already a dependency of ``anthropic``); no API key needed
for a local server (``OPENAI_API_KEY`` is sent if set).
"""
from __future__ import annotations

import json
import os
import re
from typing import List, Optional

from fedauditor.agent import SUBMIT_TOOL, SYSTEM, AgentOutcome
from fedauditor.report import validate_report
from fedauditor.tools import TOOL_SPECS, AuditSession

DEFAULT_BASE_URL = "http://localhost:11434/v1"
THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.S)
TEXT_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
SUBMIT_NOW = ("Call submit_report now with your findings (an empty findings list if the federation "
              "looks clean). Do not call other tools.")


def openai_tools() -> List[dict]:
    """The audit tools plus submit_report in OpenAI function-calling format."""
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                              "parameters": t["input_schema"]}}
            for t in TOOL_SPECS + [SUBMIT_TOOL]]


def _text_calls(content: str) -> List[dict]:
    """Tool calls written into the message text (Hermes / Qwen format)."""
    calls = []
    for i, m in enumerate(TEXT_CALL_RE.finditer(content or "")):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "name" in obj:
            args = obj.get("arguments", obj.get("parameters", {}))
            calls.append({"id": f"text_call_{i}", "type": "function",
                          "function": {"name": obj["name"],
                                       "arguments": args if isinstance(args, str) else json.dumps(args)}})
    return calls


def _text_report(content: str) -> Optional[dict]:
    """A report written as a JSON object in the text instead of a submit_report call."""
    if not content or "findings" not in content:
        return None
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", content):
        try:
            obj, _ = dec.raw_decode(content[m.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("findings"), list):
            return obj
        if isinstance(obj, dict) and isinstance(obj.get("arguments"), dict) \
                and isinstance(obj["arguments"].get("findings"), list):
            return obj["arguments"]
    return None


def _parse_args(raw) -> tuple:
    """(dict, error) from a tool call's arguments (a JSON string, or already a dict)."""
    if isinstance(raw, dict):
        return raw, None
    if raw is None or str(raw).strip() == "":
        return {}, None
    try:
        out = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, f"arguments are not valid JSON ({e.msg} at char {e.pos})"
    if not isinstance(out, dict):
        return None, "arguments must be a JSON object"
    return out, None


class OpenAIAuditor:
    """Manual tool-use loop over an OpenAI-compatible chat completions endpoint."""

    def __init__(self, model: str, base_url: str = DEFAULT_BASE_URL, api_key: Optional[str] = None,
                 max_turns: int = 40, max_tokens: int = 4096, temperature: float = 0.2,
                 thinking: bool = False, timeout: float = 900.0, max_context_chars: int = 90_000,
                 transcript_dir: Optional[str] = None, label: Optional[str] = None, verbose: bool = True):
        self.model, self.base_url = model, base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        self.max_turns, self.max_tokens, self.temperature = max_turns, max_tokens, temperature
        self.thinking, self.timeout, self.verbose = thinking, timeout, verbose
        self.max_context_chars = max_context_chars
        self.transcript_dir = transcript_dir or os.environ.get("FEDAUDITOR_TRANSCRIPT_DIR")
        self.label = label or re.sub(r"[^a-z0-9.]+", "-", model.split("/")[-1].lower()).strip("-")
        self.tools = openai_tools()

    def _create(self, messages: List[dict]) -> dict:
        import httpx
        body = {"model": self.model, "messages": messages, "tools": self.tools, "tool_choice": "auto",
                "max_tokens": self.max_tokens, "temperature": self.temperature,
                # Qwen3 chat template switch; other servers ignore unknown fields.
                "chat_template_kwargs": {"enable_thinking": self.thinking}}
        r = httpx.post(f"{self.base_url}/chat/completions", json=body, timeout=self.timeout,
                       headers={"Authorization": f"Bearer {self.api_key}"})
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
        return r.json()

    def _save(self, session: AuditSession, messages: List[dict], out: AgentOutcome) -> None:
        if not self.transcript_dir:
            return
        os.makedirs(self.transcript_dir, exist_ok=True)
        path = os.path.join(self.transcript_dir, f"{session.run.scenario.id}__{self.label}__transcript.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"stop": out.stop, "turns": out.turns, "errors": out.errors, "messages": messages},
                      fh, indent=1, default=str)

    def run(self, session: AuditSession) -> AgentOutcome:
        messages: List[dict] = []
        out = AgentOutcome(None, "max_turns", usage={"input": 0, "output": 0})
        try:
            self._loop(session, messages, out)
        finally:
            self._save(session, messages, out)
        return out

    def _submit(self, args, session, out, turn) -> Optional[str]:
        """Accept a report (returns None) or return the validation error text."""
        errs = validate_report(args, session.tools.names, session.run.rounds)
        if errs:
            out.errors.extend(errs)
            return "Invalid report: " + "; ".join(errs[:10])
        out.report, out.stop = args, "submitted"
        if self.verbose:
            print(f"  turn {turn}: submit_report ({len(args['findings'])} findings)")
        return None

    def _loop(self, session: AuditSession, messages: List[dict], out: AgentOutcome) -> None:
        system = SYSTEM
        if not self.thinking and "qwen3" in self.model.lower():
            system += "\n/no_think"   # Qwen3 soft switch; works where chat_template_kwargs is ignored (Ollama)
        messages += [{"role": "system", "content": system},
                     {"role": "user", "content": f"Audit this federation. Privacy budget: "
                                                 f"{session.accountant.total} epsilon. Begin."}]
        usage, nudges, told_to_submit = out.usage, 0, False
        for turn in range(1, self.max_turns + 1):
            out.turns = turn
            try:
                resp = self._create(messages)
            except Exception as e:  # server down, context overflow, timeout, HTTP 4xx/5xx
                out.stop = f"api_error: {type(e).__name__}: {str(e)[:300]}"
                out.errors.append(out.stop)
                return
            u = resp.get("usage") or {}
            usage["input"] += u.get("prompt_tokens", 0) or 0
            usage["output"] += u.get("completion_tokens", 0) or 0
            choice = resp["choices"][0]
            msg = choice.get("message") or {}
            content = THINK_RE.sub("", msg.get("content") or "").strip()
            calls = msg.get("tool_calls") or _text_calls(content)
            if calls and not msg.get("tool_calls"):
                content = TEXT_CALL_RE.sub("", content).strip()

            # Echo only well-formed calls; bad arguments are answered with an error below.
            parsed = []
            for i, c in enumerate(calls):
                fn = c.get("function") or {}
                args, err = _parse_args(fn.get("arguments"))
                parsed.append((c.get("id") or f"call_{turn}_{i}", fn.get("name", ""), args, err))
            assistant = {"role": "assistant", "content": content}
            if parsed:
                assistant["tool_calls"] = [{"id": cid, "type": "function",
                                            "function": {"name": name, "arguments": json.dumps(args or {})}}
                                           for cid, name, args, _ in parsed]
            messages.append(assistant)

            if not parsed:
                report = _text_report(content)
                if report is not None:
                    err = self._submit(report, session, out, turn)
                    if err is None:
                        out.stop = "submitted_text"
                        return
                    messages.append({"role": "user", "content": err + " " + SUBMIT_NOW})
                    continue
                if choice.get("finish_reason") == "length":
                    messages.append({"role": "user", "content": "Your reply was cut off. Be brief. " + SUBMIT_NOW})
                    continue
                if nudges >= 2:
                    out.stop = "no_report"
                    return
                nudges += 1
                messages.append({"role": "user", "content": "Use the tool-calling interface: call a tool, "
                                                            "or finish by calling submit_report."})
                continue

            for cid, name, args, err in parsed:
                if err:
                    out.errors.append(f"{name}: {err}")
                    messages.append({"role": "tool", "tool_call_id": cid, "content": f"Error: {err}"})
                    continue
                if name == "submit_report":
                    err = self._submit(args, session, out, turn)
                    if err is None:
                        return
                    messages.append({"role": "tool", "tool_call_id": cid, "content": err})
                    continue
                res = session.call(name, args)
                if self.verbose:
                    print(f"  turn {turn}: {res['call_id']} {name} {json.dumps(args)}"
                          f"{' ERROR ' + res['error'] if 'error' in res else ''}"
                          f" (budget left {res['budget_left']})")
                messages.append({"role": "tool", "tool_call_id": cid, "content": AuditSession.render(res)})

            # Steer towards a report before the budget, context window or turns run out.
            chars = sum(len(json.dumps(m)) for m in messages)
            reason = None
            if session.accountant.remaining < 0.05:
                reason = "The privacy budget is spent."
            elif chars > self.max_context_chars:
                reason = "The conversation is near the context limit."
            elif turn >= self.max_turns - 3:
                reason = "You are almost out of turns."
            if reason and not told_to_submit:
                told_to_submit = True
                messages.append({"role": "user", "content": f"{reason} {SUBMIT_NOW}"})
