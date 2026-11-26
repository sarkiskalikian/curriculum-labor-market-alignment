"""LLM-assisted extraction and canonicalization (OpenAI API).

The API key is read from the OPENAI_API_KEY environment variable; nothing is
ever stored in the code or notebooks.

* ``extract_skills``  - raw skill phrases from one job posting
* ``extract_topics``  - 8-15 primary academic topics from one syllabus
* ``canonicalize``    - map a raw skill or topic phrase to a short standard label,
  so that "MS Excel", "Microsoft Excel" and "Excel skills" all become "Excel"
"""

from __future__ import annotations

import json
import re
import time

import pandas as pd

from .prompts import SKILL_CANONICALIZATION, SKILL_EXTRACTION, TOPIC_CANONICALIZATION, TOPIC_EXTRACTION

EXTRACTION_MODEL = "gpt-4.1-mini"
CANONICALIZATION_MODEL = "gpt-4o"

_SKILLS_SCHEMA = {
    "type": "json_schema",
    "name": "job_skills_extraction",
    "schema": {
        "type": "object",
        "properties": {"skills": {"type": "array", "items": {"type": "string"}}},
        "required": ["skills"],
        "additionalProperties": False,
    },
}


def get_client():
    from openai import OpenAI
    return OpenAI()  # picks up OPENAI_API_KEY from the environment


def _clean_label(label: str) -> str:
    return re.sub(r"\s+", " ", str(label)).strip(" -•*.;:,")


def _unique(labels) -> list[str]:
    seen, out = set(), []
    for label in map(_clean_label, labels):
        if label and label.lower() not in seen:
            seen.add(label.lower())
            out.append(label)
    return out


def _json_call(client, model: str, system: str, user: str, retries: int = 4, wait: float = 2.0) -> dict | None:
    """One responses-API call that must return a JSON object; retried with backoff."""
    for attempt in range(1, retries + 1):
        try:
            response = client.responses.create(
                model=model,
                input=[{"role": "system", "content": [{"type": "input_text", "text": system}]},
                       {"role": "user", "content": [{"type": "input_text", "text": user}]}],
                text={"format": {"type": "json_object"}},
            )
            data = json.loads(response.output_text.strip())
            if isinstance(data, dict):
                return data
        except Exception:
            time.sleep(wait * attempt)
    return None


def extract_skills(client, text: str, model: str = EXTRACTION_MODEL) -> tuple[list[str], str]:
    """Skills mentioned in one job posting. Returns (skills, status)."""
    if pd.isna(text) or len(str(text).strip()) < 30:
        return [], "too_short"
    try:
        response = client.responses.create(
            model=model,
            input=[{"role": "system", "content": SKILL_EXTRACTION},
                   {"role": "user", "content": str(text)[:6000]}],
            text={"format": _SKILLS_SCHEMA},
        )
        skills = json.loads(response.output_text).get("skills", [])
        return [str(s).strip() for s in skills if str(s).strip()], "success"
    except Exception as e:  # keep going; failures are logged per row
        return [], f"error: {e}"


def extract_topics(client, course_code: str, course_title: str, content: str,
                   model: str = EXTRACTION_MODEL) -> tuple[list[str], str]:
    """Primary academic topics of one course. Returns (topics, status)."""
    prompt = f"Course Code: {course_code}\nCourse Title: {course_title}\n\nCourse Content:\n{content}\n"
    data = _json_call(client, model, TOPIC_EXTRACTION, prompt)
    if data is None or not isinstance(data.get("primary_topics"), list):
        return [], "failed"
    return _unique(t for t in data["primary_topics"] if t is not None), "success"


def canonicalize(client, phrase: str, kind: str = "skill", model: str = CANONICALIZATION_MODEL) -> str:
    """Map one raw phrase to its canonical label. kind is 'skill' or 'topic'."""
    system, key = ((SKILL_CANONICALIZATION, "canonical_skill") if kind == "skill"
                   else (TOPIC_CANONICALIZATION, "canonical_topic"))
    data = _json_call(client, model, system, f"Raw {kind}: {phrase}")
    label = _clean_label(data.get(key, "")) if data else ""
    return label or _clean_label(phrase)  # keep the raw phrase rather than dropping it


def map_pipe(pipe_text, mapping: dict[str, str]) -> list[str]:
    """Replace each raw item in 'a | b | c' with its canonical label, keeping order and removing duplicates."""
    if pd.isna(pipe_text):
        return []
    seen, out = set(), []
    for raw in (t.strip() for t in str(pipe_text).split("|")):
        if not raw:
            continue
        label = str(mapping.get(raw, raw)).strip()
        if label and label.lower() not in seen:
            seen.add(label.lower())
            out.append(label)
    return out
