from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .schemas import JudgeResult, QAExample, ReflectionEntry


DEFAULT_GEMINI_MODEL = "gemma-4-31b-it"


@lru_cache(maxsize=1)
def _client() -> Any:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY before using Gemini runtime.")

    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError("Install Gemini SDK first: pip install google-genai") from exc

    return genai.Client(api_key=api_key)


def _usage_tokens(response: Any) -> int:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return 0

    total = _usage_value(usage, "total_token_count")
    if isinstance(total, int):
        return total

    prompt = _usage_value(usage, "prompt_token_count") or 0
    candidates = _usage_value(usage, "candidates_token_count") or 0
    return prompt + candidates


def _usage_value(usage: Any, key: str) -> int | None:
    value = usage.get(key) if isinstance(usage, dict) else getattr(usage, key, None)
    return value if isinstance(value, int) else None


def _generate(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str,
    response_mime_type: str = "text/plain",
) -> tuple[str, int]:
    client = _client()
    from google.genai import types

    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.0,
            response_mime_type=response_mime_type,
        ),
    )
    text = getattr(response, "text", "") or ""
    return text.strip(), _usage_tokens(response)


def _context_text(example: QAExample) -> str:
    return "\n\n".join(f"[{chunk.title}]\n{chunk.text}" for chunk in example.context)


def _json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()

    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        value = json.loads(text[start : end + 1])

    if not isinstance(value, dict):
        raise ValueError("Gemini response must be a JSON object.")
    return value


def actor_answer(
    example: QAExample,
    attempt_id: int,
    agent_type: str,
    reflection_memory: list[str],
    *,
    model: str = DEFAULT_GEMINI_MODEL,
) -> tuple[str, int]:
    memory = "\n".join(f"- {item}" for item in reflection_memory) or "(none)"
    user_prompt = f"""Question:
                    {example.question}

                    Context:
                    {_context_text(example)}

                    Agent type: {agent_type}
                    Attempt: {attempt_id}

                    Reflection memory:
                    {memory}
                    """
    return _generate(ACTOR_SYSTEM, user_prompt, model=model)


def evaluator(
    example: QAExample,
    answer: str,
    *,
    model: str = DEFAULT_GEMINI_MODEL,
) -> tuple[JudgeResult, int]:
    user_prompt = f"""Question:
                    {example.question}

                    Context:
                    {_context_text(example)}

                    Gold answer:
                    {example.gold_answer}

                    Predicted answer:
                    {answer}
                    """
    text, tokens = _generate(
        EVALUATOR_SYSTEM,
        user_prompt,
        model=model,
        response_mime_type="application/json",
    )
    return JudgeResult.model_validate(_json_object(text)), tokens


def reflector(
    example: QAExample,
    attempt_id: int,
    judge: JudgeResult,
    *,
    model: str = DEFAULT_GEMINI_MODEL,
) -> tuple[ReflectionEntry, int]:
    user_prompt = f"""Question:
                    {example.question}

                    Context:
                    {_context_text(example)}

                    Failed attempt id:
                    {attempt_id}

                    Failure reason:
                    {judge.reason}

                    Missing evidence:
                    {judge.missing_evidence}

                    Spurious claims:
                    {judge.spurious_claims}
                    """
    text, tokens = _generate(
        REFLECTOR_SYSTEM,
        user_prompt,
        model=model,
        response_mime_type="application/json",
    )
    payload = _json_object(text)
    payload.setdefault("attempt_id", attempt_id)
    return ReflectionEntry.model_validate(payload), tokens
