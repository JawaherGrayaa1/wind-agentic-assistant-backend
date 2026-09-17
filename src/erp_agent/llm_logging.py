from __future__ import annotations

import json
import sys
import time
from typing import Any


def _safe_print(text: str) -> None:
    """Safely print text to stdout even on Windows consoles with cp1252 encoding."""
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        # Fallback to ascii/utf-8 buffer write
        sys.stdout.buffer.write((text + "\n").encode("utf-8", errors="replace"))
        sys.stdout.buffer.flush()


def get_location_label(url: str) -> str:
    """Determine whether the URL points to a local or remote (Kaggle/ngrok) instance."""
    if "127.0.0.1" in url or "localhost" in url:
        return "LOCAL [Ollama on this machine]"
    elif "ngrok" in url:
        return "REMOTE [Kaggle GPU via ngrok tunnel]"
    return f"REMOTE [{url}]"


def log_llm_request(component: str, url: str, model: str, input_summary: str) -> float:
    """Log outgoing LLM request and return start time."""
    location = get_location_label(url)
    _safe_print("\n" + "=" * 65)
    _safe_print(f"[LLM REQUEST] -> {component}")
    _safe_print(f"  * Mode / Target:  {location}")
    _safe_print(f"  * Endpoint:       {url}")
    _safe_print(f"  * Model:          {model}")
    _safe_print(f"  * Prompt Input:   {input_summary}")
    _safe_print("-" * 65)
    return time.time()


def log_llm_response(component: str, response_data: Any, start_time: float | None = None) -> None:
    """Log received LLM response."""
    elapsed = f" (took {time.time() - start_time:.2f}s)" if start_time else ""
    _safe_print(f"[LLM RESPONSE]{elapsed} <- {component}:")
    if isinstance(response_data, dict):
        _safe_print(json.dumps(response_data, indent=2, ensure_ascii=False))
    elif hasattr(response_data, "__dict__"):
        _safe_print(json.dumps(response_data.__dict__, indent=2, ensure_ascii=False, default=str))
    else:
        _safe_print(str(response_data))
    _safe_print("=" * 65 + "\n")


def log_llm_error(component: str, url: str, error: Exception) -> None:
    """Log an LLM error or unreachable server."""
    location = get_location_label(url)
    _safe_print("\n" + "!" * 65)
    _safe_print(f"[LLM ERROR] in {component}")
    _safe_print(f"  * Target:  {location} ({url})")
    _safe_print(f"  * Details: {error}")
    _safe_print("!" * 65 + "\n")


def log_rule_planner(input_text: str, result_plan: Any) -> None:
    """Log when RulePlanner is used instead of LLM."""
    _safe_print("\n" + "-" * 65)
    _safe_print("[RULE PLANNER] (Local hardcoded heuristic rules)")
    _safe_print(f"  * Input:  {input_text}")
    _safe_print(f"  * Output: Intent='{result_plan.intent}', Reply='{result_plan.reply}', Tool='{result_plan.tool_name}'")
    _safe_print("-" * 65 + "\n")
