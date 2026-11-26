"""Normalize the structured syllabus tables into one text per course.

The syllabi (PDF / DOCX / PPTX) were first structured into two tables, one per
college, with a column per syllabus section. This module keeps the three
sections that describe *what is taught* (description, topics, learning
outcomes), cleans them, drops administrative lines ("Midterm", "Quiz",
"Spring break"), merges duplicate rows of the same course, and builds the
``course_content_for_extraction`` text that the topic extractor reads.

Instructor names, emails and policies in the source tables are never used.
"""

from __future__ import annotations

import ast
import html
import json
import re
import unicodedata

import numpy as np
import pandas as pd

from .text import is_missing

CONTENT_COLUMNS = ["course_description", "topics", "course_learning_outcomes"]

_REPLACEMENTS = {
    " ": " ", "​": " ", "•": "\n- ", "": "\n- ", "": "\n- ", "·": "\n- ",
    "â€“": "-", "â€”": "-", "â€˜": "'", "â€™": "'", "â€œ": '"', "â€\x9d": '"',
    "â€¦": "...", "âˆ’": "-", "–": "-", "—": "-", "’": "'", "“": '"', "”": '"',
}

ADMIN_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"^final exam$", r"^midterm$", r"^midterm exam$", r"^quiz$", r"^quizzes$",
    r"^review$", r"^review syllabus$", r"^course overview$", r"^course introduction$",
    r"^introduction of the lecturer$", r"^introduction to course syllables$",
    r"^introduction to course syllabus$", r"^grading due$", r"^spring break$",
    r"^holiday$", r"^off$", r"^q&a$", r"^group project presentations$",
]]


def repair_mojibake(text) -> str:
    if is_missing(text):
        return ""
    s = html.unescape(str(text))
    if any(tok in s for tok in ("â", "Ã", "Â", "\x95", "\x96", "\x97")):
        for enc in ("latin1", "cp1252"):
            try:
                repaired = s.encode(enc, errors="ignore").decode("utf-8", errors="ignore")
            except Exception:
                continue
            if repaired and repaired.count("â") + repaired.count("Ã") < s.count("â") + s.count("Ã"):
                s = repaired
                break
    for bad, good in _REPLACEMENTS.items():
        s = s.replace(bad, good)
    return unicodedata.normalize("NFKC", s)


def _unique(items) -> list[str]:
    seen, out = set(), []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


def _topics_from_struct(obj) -> list[str]:
    """Some topic cells are JSON lists of {week, topic} objects; pull out the topic strings."""
    found = []
    if isinstance(obj, dict):
        for key in ("topic", "topics", "module", "content", "title", "subject"):
            if key in obj and not is_missing(obj[key]):
                found.append(str(obj[key]))
        for v in obj.values():
            if isinstance(v, (dict, list)):
                found.extend(_topics_from_struct(v))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(_topics_from_struct(item))
    return found


def _parse_topics_cell(cell) -> str:
    s = repair_mojibake(cell).strip()
    if s.startswith(("[", "{")):
        for parser in (json.loads, ast.literal_eval):
            try:
                topics = _unique(t for t in _topics_from_struct(parser(s)) if not is_missing(t))
                return " | ".join(topics)
            except Exception:
                pass
    return s


def _is_admin_line(line: str) -> bool:
    x = re.sub(r"^[\-\*\•\d\.\)\(:\s]+", "", line.strip())
    x = re.sub(r"\s+", " ", x)
    if not x:
        return True
    if any(p.fullmatch(x) for p in ADMIN_PATTERNS):
        return True
    if re.fullmatch(r"(week|session|topic|chapter|ch|hw)\s*[\#:\-]?\s*\d+.*", x, flags=re.IGNORECASE):
        return True
    # short labels like "Week 3", "Feb 12", "3/4" carry no topic information
    if re.fullmatch(r"[\d\s,/\-:A-Za-z]{1,20}", x) and len(re.findall(r"[A-Za-z]", x)) <= 3:
        return True
    return False


def clean_field(text, field: str = "generic") -> str:
    """Split a syllabus section into lines, drop admin-only lines and duplicates, join with ' | '."""
    if is_missing(text):
        return ""
    s = _parse_topics_cell(text) if field == "topics" else repair_mojibake(text)
    s = s.replace("\\n", "\n")
    s = re.sub(r"\s+\|\s+", "\n", s)
    s = re.sub(r"\s*;\s*", "\n", s)
    s = re.sub(r"\t+", " ", s)
    s = re.sub(r"[ ]{2,}", " ", s)
    s = re.sub(r"\n{2,}", "\n", s).strip()

    lines = [part.strip(" -•*:\t") for part in s.split("\n")]
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in lines if ln.strip(" -•*:\t")]
    return " | ".join(_unique(ln for ln in lines if ln and not _is_admin_line(ln)))


def _course_key(code: str, title: str) -> str:
    code_key = re.sub(r"\s+", "", str(code).upper().strip())
    if code_key:
        return code_key
    title_key = repair_mojibake(title).lower().strip()
    title_key = re.sub(r"[^a-z0-9\s/&\-]", "", re.sub(r"\s+", " ", title_key))
    return "TITLE::" + title_key


def _content_text(title, description, topics, outcomes) -> str:
    parts = [("Course Title", title), ("Course Description", description),
             ("Topics", topics), ("Course Learning Outcomes", outcomes)]
    return "\n".join(f"{label}: {value}" for label, value in parts if value).strip()


def normalize_syllabi(bus: pd.DataFrame, cse: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (row-level table, course-level table) ready for topic extraction."""
    frames = []
    for df, name in ((bus, "BUS"), (cse, "CSE")):
        df = df.copy()
        df["source_dataset"] = name
        for col in ["filename", "course_code", "course_title", *CONTENT_COLUMNS]:
            if col not in df.columns:
                df[col] = np.nan
        frames.append(df[["source_dataset", "filename", "course_code", "course_title", *CONTENT_COLUMNS]])
    rows = pd.concat(frames, ignore_index=True)

    rows["course_code_clean"] = rows["course_code"].fillna("").astype(str).str.strip()
    rows["course_title_clean"] = rows["course_title"].fillna("").astype(str).apply(repair_mojibake).str.strip()
    for col in CONTENT_COLUMNS:
        rows[f"{col}_norm"] = rows[col].apply(lambda x, c=col: clean_field(x, "topics" if c == "topics" else c))
    rows["course_key"] = [_course_key(c, t) for c, t in zip(rows.course_code_clean, rows.course_title_clean)]

    def longest(series):
        vals = [str(x).strip() for x in series if not is_missing(x)]
        return sorted(vals, key=lambda s: (len(s), s), reverse=True)[0] if vals else ""

    def merge(series):
        return " || ".join(_unique(str(x).strip() for x in series if not is_missing(x)))

    courses = (rows.groupby("course_key", dropna=False)
                   .agg(source_dataset=("source_dataset", lambda s: " | ".join(_unique(map(str, s)))),
                        row_count=("course_key", "size"),
                        course_code=("course_code_clean", longest),
                        course_title=("course_title_clean", longest),
                        course_description_norm=("course_description_norm", merge),
                        topics_norm=("topics_norm", merge),
                        course_learning_outcomes_norm=("course_learning_outcomes_norm", merge))
                   .reset_index())
    courses["course_content_for_extraction"] = [
        _content_text(*vals) for vals in courses[["course_title", "course_description_norm", "topics_norm",
                                                   "course_learning_outcomes_norm"]].itertuples(index=False)]
    courses = courses.sort_values(["course_title", "course_code"]).reset_index(drop=True)
    return rows, courses
