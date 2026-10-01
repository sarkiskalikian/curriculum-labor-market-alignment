"""Merge the per-category scraper exports into one clean job-posting corpus.

Each scraper run produces one CSV per category and platform, already translated
to English. Column names differ slightly between staff.am and job.am, so every
file is mapped onto a common schema, cleaned, de-duplicated, and reduced to a
single ``job_content_for_extraction`` field (title + description +
responsibilities + requirements) that is passed to the skill extractor.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .text import normalize_long_text, normalize_short_text

TITLE_CANDS = ["title_en", "job_title_en", "job_title", "title", "position", "vacancy_title"]
COMPANY_CANDS = ["company_name", "company", "employer", "organization"]
URL_CANDS = ["job_url", "url", "link", "vacancy_url"]
DESC_CANDS = ["full_description_en", "description_en", "job_description_en",
              "full_description", "description", "job_description"]
RESP_CANDS = ["responsibilities_en", "responsibility_en", "responsibilities",
              "responsibility", "duties_en", "duties"]
REQ_CANDS = ["requirements_en", "requirement_en", "requirements", "requirement",
             "qualifications_en", "qualifications", "skills_en", "skills"]
INDUSTRY_CANDS = ["industry", "category", "sector"]


def read_csv_any_encoding(path: str | Path) -> pd.DataFrame:
    last_error = None
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, pd.errors.ParserError) as e:
            last_error = e
    raise last_error


def platform_and_category(filename: str) -> tuple[str, str]:
    """'staffam_finance_jobs_complete__translated_EN.csv' -> ('staffam', 'finance')."""
    name = Path(filename).stem.lower()
    name = re.sub(r"_+translated_en$", "", name)
    platform = next((p for p in ("jobam", "staffam") if name.startswith(p + "_")), "unknown")
    category = name[len(platform) + 1:] if platform != "unknown" else name
    category = category.replace("_jobs_complete", "").replace("__", "_").strip("_ ")
    return platform, category


def _first_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {c.lower().strip(): c for c in df.columns}
    return next((lookup[c] for c in candidates if c in lookup), None)


def standardize(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    platform, category = platform_and_category(filename)

    def col(cands):
        name = _first_column(df, cands)
        return df[name] if name else pd.Series([""] * len(df))

    out = pd.DataFrame({
        "source_filename": filename,
        "source_platform": platform,
        "source_category": category,
        "job_title": col(TITLE_CANDS).apply(normalize_short_text),
        "company_name": col(COMPANY_CANDS).apply(normalize_short_text),
        "job_url": col(URL_CANDS).apply(normalize_short_text),
        "industry_clean": col(INDUSTRY_CANDS).apply(normalize_short_text),
        "description_clean": col(DESC_CANDS).apply(normalize_long_text),
        "responsibilities_clean": col(RESP_CANDS).apply(normalize_long_text),
        "requirements_clean": col(REQ_CANDS).apply(normalize_long_text),
    })

    def content(row):
        parts = [("Job Title", row.job_title), ("Description", row.description_clean),
                 ("Responsibilities", row.responsibilities_clean), ("Requirements", row.requirements_clean)]
        return "\n".join(f"{label}: {text}" for label, text in parts if text).strip()

    out["job_content_for_extraction"] = out.apply(content, axis=1)
    out["content_length_chars"] = out["job_content_for_extraction"].str.len()
    return out


def _key(text) -> str:
    return re.sub(r"\s+", " ", str(text).lower().strip()) if pd.notna(text) else ""


def build_corpus(csv_paths: list[str | Path]) -> pd.DataFrame:
    """Merge all exports, drop duplicates (same URL, or same platform+title+company), drop empty rows."""
    frames = [standardize(read_csv_any_encoding(p), Path(p).name) for p in sorted(csv_paths)]
    df = pd.concat(frames, ignore_index=True)

    url_key = df["job_url"].apply(_key)
    fallback_key = (df["source_platform"] + "|||" + df["job_title"].apply(_key)
                    + "|||" + df["company_name"].apply(_key))
    df["dedup_key"] = url_key.where(url_key != "", fallback_key)

    # When a posting appears twice, keep the copy with the most text.
    df = (df.sort_values("content_length_chars", ascending=False)
            .drop_duplicates("dedup_key", keep="first")
            .reset_index(drop=True))

    keep = (df["job_title"].str.len() > 0) | (df["job_content_for_extraction"].str.len() >= 40)
    return df[keep].reset_index(drop=True)
