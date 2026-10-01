"""Human-in-the-loop validation layer.

The hybrid matcher gives a binary answer (covered / missing). That is useful
but blunt: a skill can be matched to a related topic without really being
taught, and a skill can be taught under a name the matcher doesn't recognise.

So every result was reviewed. An LLM (Claude Sonnet 4.6) read the candidate
courses' content and suggested COVERED / PARTIAL / MISSING with a one-sentence
reason (see ``review_skill``); I then checked each suggestion and assigned a
final gap type. The final status follows directly from that gap type
(``GAP_TYPE_TO_STATUS``).
"""

from __future__ import annotations

import json
import re

import pandas as pd

from .prompts import COVERAGE_REVIEW

GAP_TYPE_TO_STATUS = {
    "Lexical Coverage": "COVERED",
    "Semantic Coverage": "COVERED",
    "Partial / Documentation Gap": "PARTIAL",
    "Teaching Gap": "MISSING",
}

STATUS_RULES = {
    "Lexical Coverage": "Direct wording/tool match; counted as covered.",
    "Semantic Coverage": "Different wording but same underlying concept; counted as covered.",
    "Partial / Documentation Gap": ("Related curriculum evidence exists, but coverage is incomplete, "
                                    "indirect, shallow, or affected by documentation/canonicalization "
                                    "ambiguity; counted as partial."),
    "Teaching Gap": "No sufficient evidence of curriculum coverage; counted as missing.",
}


def apply_validation(results: pd.DataFrame, gap_col: str = "gap_type") -> pd.DataFrame:
    """Add the final 3-class status and the rule behind it, given reviewed gap types."""
    out = results.copy()
    out["final_status"] = out[gap_col].map(GAP_TYPE_TO_STATUS)
    out["final_rule"] = out[gap_col].map(STATUS_RULES)
    unknown = out["final_status"].isna()
    if unknown.any():
        raise ValueError(f"Unknown gap types: {sorted(out.loc[unknown, gap_col].unique())}")
    return out


def _parse_json(text: str) -> dict | None:
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


def candidate_courses(skill: str, courses: dict[str, dict], best_topic: str = "",
                      covered_by: str = "", top_n: int = 5) -> list[str]:
    """Rank courses worth showing the reviewer, using simple keyword scoring.

    ``courses`` maps course code -> {"title", "topics" (list), "text" (syllabus text)}.
    """
    stop = {"and", "the", "of", "for", "in", "a", "an", "to", "with", "or", "is", "are", "&"}
    words = set(re.sub(r"[-/.+]", " ", skill.lower()).split()) - stop
    words = {w for w in words if len(w) > 2}
    best_words = {w for w in str(best_topic).lower().split() if len(w) > 2} - stop

    scores = {}
    for code, info in courses.items():
        topics = " ".join(info["topics"]).lower()
        content = info["text"][:1500].lower()
        score = sum(3 if w in topics else 1 if w in content else 0 for w in words)
        score += 15 if skill.lower() in topics else 8 if skill.lower() in content else 0
        score += sum(4 for w in best_words if w in topics)
        score += 25 if code in str(covered_by) else 0
        scores[code] = score
    return [code for code, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:top_n]]


def review_skill(client, skill: str, domain: str, n_jobs: int, courses: dict[str, dict],
                 model_status: str, model_score: float, best_topic: str, covered_by: str = "",
                 model: str = "claude-sonnet-4-6") -> dict:
    """Ask the LLM reviewer for COVERED / PARTIAL / MISSING plus a one-line reason.

    ``client`` is an ``anthropic.Anthropic()`` instance (reads ANTHROPIC_API_KEY).
    """
    block = ""
    for code in candidate_courses(skill, courses, best_topic, covered_by):
        info = courses[code]
        block += (f"\n---\nCOURSE: {code} | {info['title']}\n"
                  f"TOPICS: {' | '.join(info['topics'][:12])}\nCONTENT: {info['text'][:800]}\n")

    if model_status == "COVERED":
        hint = (f"\nNote: A previous embedding model matched this skill to the topic "
                f"'{best_topic}' (score: {model_score:.2f}). Verify if this is a genuine match.")
    else:
        hint = "\nNote: A previous embedding model found no match for this skill. Confirm or find a match if one exists."

    prompt = COVERAGE_REVIEW.format(skill=skill, domain=domain, n_jobs=n_jobs,
                                    model_hint=hint, course_block=block)
    try:
        msg = client.messages.create(model=model, max_tokens=300,
                                     messages=[{"role": "user", "content": prompt}])
        parsed = _parse_json(msg.content[0].text)
        if parsed and "status" in parsed:
            return parsed
        return {"status": "ERROR", "covered_by": "", "reasoning": "could not parse response"}
    except Exception as e:
        return {"status": "ERROR", "covered_by": "", "reasoning": str(e)[:80]}
