"""Small text helpers shared by the scrapers and the corpus builder."""

from __future__ import annotations

import html
import re
import unicodedata

import pandas as pd

# Mojibake and typographic characters that show up when job boards are
# exported through different encodings.
_REPLACEMENTS = {
    " ": " ",
    "​": " ",
    "•": "\n- ",
    "": "\n- ",
    "·": "\n- ",
    "â€“": "-",
    "â€”": "-",
    "â€˜": "'",
    "â€™": "'",
    "â€œ": '"',
    "â€\x9d": '"',
    "â€¦": "...",
    "âˆ’": "-",
    "–": "-",
    "—": "-",
    "’": "'",
    "“": '"',
    "”": '"',
}


def is_missing(x) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and pd.isna(x):
        return True
    s = str(x).strip()
    return s == "" or s.lower() in {"nan", "none", "null", "[]", "{}"}


def clean_html(html_text) -> str:
    """Strip tags and decode the handful of entities staff.am returns."""
    if not html_text:
        return ""
    s = re.sub(r"<[^<]+?>", "", str(html_text))
    s = re.sub(r"\s+", " ", s).strip()
    return html.unescape(s)


def fix_text(text) -> str:
    """Unescape, repair common mojibake, and normalise whitespace (keeps line breaks)."""
    if is_missing(text):
        return ""
    s = html.unescape(str(text))
    for bad, good in _REPLACEMENTS.items():
        s = s.replace(bad, good)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\\n", "\n")
    s = re.sub(r"\r\n?", "\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def normalize_short_text(text) -> str:
    """Single-line fields such as titles and company names."""
    return re.sub(r"\s+", " ", fix_text(text)).strip()


def normalize_long_text(text) -> str:
    """Multi-line fields (description, requirements) joined as 'line | line | line'."""
    lines = [ln.strip(" -•*:\t") for ln in fix_text(text).split("\n")]
    return " | ".join(ln for ln in lines if ln)


def detect_language(text: str) -> str:
    """Rough script-based language guess: 'am' (Armenian), 'ru' (Cyrillic), 'en' (Latin)."""
    if not text or len(text.strip()) < 3:
        return "unknown"
    armenian = len(re.findall(r"[԰-֏]", text))
    cyrillic = len(re.findall(r"[Ѐ-ӿ]", text))
    latin = len(re.findall(r"[a-zA-Z]", text))
    total = armenian + cyrillic + latin
    if total == 0:
        return "unknown"
    if armenian / total >= 0.25:
        return "am"
    if cyrillic / total >= 0.25:
        return "ru"
    if latin / total >= 0.25:
        return "en"
    return "unknown"
