"""Gemini tutor logic for aromatic synthesis practice.

The functions in this module keep the answer key and route state hidden from
students while asking Gemini to coach with Socratic prompts.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from typing import Any, Optional

from google import genai
from google.genai import types


MODEL_NAME = "gemini-2.5-flash"
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 2.0
RETRYABLE_STATUS_CODES = {429, 503}


def get_synthesis_tutor_response(
    chat_history: Sequence[Mapping[str, Any]],
    current_route_data: Mapping[str, Any],
) -> str:
    """Return Socratic synthesis coaching for a target-driven route.

    Args:
        chat_history: Chronological chat messages. The latest student message
            should already be included by the caller when the student types one.
        current_route_data: Hidden state containing benzene as the starting
            material, target name/SMILES, reagent sequence so far, current
            intermediate, last engine outcome, and optional recommended route.
    """

    return _generate_with_retry(
        chat_history=chat_history,
        system_instruction=_build_synthesis_system_instruction(current_route_data),
    )


def get_single_step_tutor_response(
    chat_history: Sequence[Mapping[str, Any]],
    problem_data: Mapping[str, Any],
) -> str:
    """Return Socratic coaching for a single-step EAS prediction problem."""

    return _generate_with_retry(
        chat_history=chat_history,
        system_instruction=_build_single_step_system_instruction(problem_data),
    )


def _generate_with_retry(
    chat_history: Sequence[Mapping[str, Any]],
    system_instruction: str,
) -> str:
    """Call Gemini with exponential backoff for temporary capacity errors."""

    contents = _to_gemini_contents(chat_history)
    delay_seconds = INITIAL_BACKOFF_SECONDS
    last_retryable_error: Optional[BaseException] = None

    for attempt_index in range(MAX_RETRIES + 1):
        try:
            with genai.Client() as client:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.35,
                        max_output_tokens=800,
                    ),
                )
            return response.text or "Let's keep reasoning from the last step."
        except Exception as exc:  # The SDK may surface provider-specific errors.
            status_code = _extract_status_code(exc)
            if status_code not in RETRYABLE_STATUS_CODES:
                return (
                    "I could not reach the synthesis coach just now. "
                    "Please check the API key and network connection, then try again."
                )

            last_retryable_error = exc
            if attempt_index >= MAX_RETRIES:
                break

            time.sleep(delay_seconds)
            delay_seconds *= 2

    return (
        "Gemini is temporarily busy, so I could not get a tutor response after "
        f"{MAX_RETRIES} retries. Your route state is still saved; try again in a "
        "moment."
    )


def _extract_status_code(exc: BaseException) -> Optional[int]:
    """Best-effort extraction of HTTP status from google-genai exceptions."""

    for attr_name in ("status_code", "code"):
        raw_value = getattr(exc, attr_name, None)
        if isinstance(raw_value, int):
            return raw_value
        if isinstance(raw_value, str) and raw_value.isdigit():
            return int(raw_value)

    message = str(exc).lower()
    if "429" in message or "resource exhausted" in message:
        return 429
    if "503" in message or "service unavailable" in message:
        return 503
    return None


def _build_synthesis_system_instruction(current_route_data: Mapping[str, Any]) -> str:
    """Build the hidden instruction for target-oriented route coaching."""

    hidden_route_json = json.dumps(current_route_data, sort_keys=True, default=str)

    return f"""
You are a Socratic Synthesis Coach for an Organic Chemistry II aromatic
synthesis playground. The student is trying to synthesize a target molecule
from benzene using undergraduate textbook reactions.

Your coaching rules are strict:
- Do not simply give the full route or exact next reagent.
- Guide the student to critique their own route using short, focused questions.
- Use the hidden route data to understand the current intermediate, target,
  reagents used so far, engine warnings, and the recommended route.
- If the student added a reagent out of order, gently identify the conflict
  using chemical theory, then ask a guiding question.
- Mention relevant theory when useful: ortho/para vs meta directors, activators
  vs deactivators, Friedel-Crafts failure on strongly deactivated rings,
  Lewis-acid complexation, benzylic oxidation, Clemmensen reduction, nitro
  reduction, diazotization, and Sandmeyer substitutions.
- If the current step failed, explain why in student-friendly language and ask
  what functional group or directing effect they should set up first.
- If the student requests a hint, give a targeted hint without revealing the
  exact next reagent.
- If the route reaches the target, congratulate the student briefly and ask them
  to explain why the order worked.
- Keep responses concise enough for a Streamlit chat panel.
- Never say that you received hidden route data, an answer key, or private state.

Hidden current route data:
{hidden_route_json}
""".strip()


def _build_single_step_system_instruction(problem_data: Mapping[str, Any]) -> str:
    """Build the hidden instruction for single-step EAS coaching."""

    hidden_problem_json = json.dumps(problem_data, sort_keys=True, default=str)

    return f"""
You are an interactive Organic Chemistry II tutor for a single-step
Electrophilic Aromatic Substitution problem.

Use the Socratic method. Do not immediately reveal the product. Walk the student
through:
Step A: Identify the electrophile made by the reagent system.
Step B: Classify the directing group and choose ortho/para or meta positions.
Step C: Explain deprotonation and restoration of aromaticity.

Ask one focused question at a time. If the student is wrong, gently point to the
specific directing or mechanism issue and ask a follow-up. Confirm the final
answer only after a serious attempt or explicit check request. Never mention
hidden problem data or an answer key.

Hidden problem data:
{hidden_problem_json}
""".strip()


def _to_gemini_contents(chat_history: Sequence[Mapping[str, Any]]) -> list[types.Content]:
    """Convert app chat dictionaries into Gemini Content objects."""

    contents: list[types.Content] = []
    for message in chat_history:
        text = _message_text(message)
        if not text:
            continue

        contents.append(
            types.Content(
                role=_normalize_role(str(message.get("role", "user"))),
                parts=[types.Part.from_text(text=text)],
            )
        )

    if not contents:
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text="Please help me think this through.")],
            )
        )
    return contents


def _normalize_role(role: str) -> str:
    """Map app roles to Gemini roles."""

    if role.strip().lower() in {"assistant", "model", "ai", "coach", "tutor"}:
        return "model"
    return "user"


def _message_text(message: Mapping[str, Any]) -> str:
    """Extract message text from common chat-history shapes."""

    raw_text = message.get("content", message.get("text", ""))
    if raw_text is None:
        return ""
    return str(raw_text).strip()
