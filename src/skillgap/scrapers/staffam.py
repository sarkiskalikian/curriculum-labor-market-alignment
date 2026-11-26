"""Scraper for staff.am job listings.

staff.am is a Next.js site, so every listing and job page ships its data as JSON
inside a ``<script id="__NEXT_DATA__">`` tag. We read that JSON instead of
parsing the rendered HTML, and only fall back to the visible "skill chips" when
the JSON has no skills for a posting.

Category IDs used in the study: 1 = Software development, 8 = Administrative,
13 = Finance (plus several others listed in the README).
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import pandas as pd
import requests
from bs4 import BeautifulSoup

from ..text import clean_html

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
}

LANGS = ("en", "am", "ru")


def _dedup_keep_order(items) -> list[str]:
    seen, out = set(), []
    for x in items:
        x = clean_html(x)
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _next_data(html_bytes: bytes) -> tuple[dict, BeautifulSoup]:
    soup = BeautifulSoup(html_bytes, "html.parser")
    tag = soup.find("script", {"id": "__NEXT_DATA__"})
    if not tag or not tag.string:
        return {}, soup
    return json.loads(tag.string).get("props", {}).get("pageProps", {}), soup


def normalize_skill_value(value) -> dict[str, str]:
    """Skills come as a {en, am, ru} dict, a list of strings, or a list of dicts. Flatten all of them."""
    empty = {lang: "" for lang in LANGS}
    if not value:
        return empty
    if isinstance(value, dict) and any(k in value for k in LANGS):
        return {lang: clean_html(value.get(lang, "")) for lang in LANGS}
    if isinstance(value, list):
        collected = {lang: [] for lang in LANGS}
        for item in value:
            if isinstance(item, str):
                collected["en"].append(item)
            elif isinstance(item, dict):
                t = item.get("title") or item.get("name") or item.get("value")
                if isinstance(t, dict):
                    for lang in LANGS:
                        collected[lang].append(t.get(lang, ""))
                elif isinstance(t, str):
                    collected["en"].append(t)
        return {lang: ", ".join(_dedup_keep_order(collected[lang])) for lang in LANGS}
    if isinstance(value, str):
        return {**empty, "en": clean_html(value)}
    return empty


def skills_from_visible_text(soup: BeautifulSoup, start_label: str) -> list[str]:
    """Read the skill chips listed under a heading such as 'Professional Skills'."""
    stop_labels = {"apply now", "about company", "benefits", "discover the employer",
                   "required candidate level", "employment term", "category"}
    lines = [clean_html(x) for x in soup.get_text("\n", strip=True).splitlines()]
    lines = [x for x in lines if x]

    start = next((i for i, line in enumerate(lines) if line.lower() == start_label.lower()), None)
    if start is None:
        return []

    skills = []
    for line in lines[start + 1:start + 40]:
        low = line.lower()
        if low in stop_labels:
            break
        if 2 <= len(line) <= 40 and low not in {"yerevan", "remote", "full time", "part time", "permanent"}:
            skills.append(line)
    return _dedup_keep_order(skills)


def _localized(job: dict, key: str) -> dict[str, str]:
    block = job.get(key) or {}
    return {lang: clean_html(block.get(lang, "")) for lang in LANGS}


def get_job_detail(job_url: str, session: requests.Session | None = None) -> dict | None:
    """Full description, responsibilities, requirements and skills for one posting."""
    session = session or requests.Session()
    try:
        response = session.get(job_url, headers=HEADERS, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"    could not fetch {job_url}: {str(e)[:120]}")
        return None

    page, soup = _next_data(response.content)
    job = page.get("job") or {}
    if not job:
        return None

    professional = normalize_skill_value(job.get("professional_skills"))
    personal = normalize_skill_value(job.get("personal_skills"))

    # Some employers write their skills inside the requirements or additional
    # info block instead of the dedicated field.
    for fallback in ("requirements", "additional_info"):
        if professional["en"]:
            break
        text = clean_html((job.get(fallback) or {}).get("en", ""))
        if "skills" in text.lower():
            professional["en"] = text

    detail = {}
    for field, out_name in [("description", "full_description"),
                            ("responsibilities", "responsibilities"),
                            ("requirements", "requirements")]:
        for lang, text in _localized(job, field).items():
            detail[f"{out_name}_{lang}"] = text
    for lang in LANGS:
        detail[f"professional_skills_{lang}"] = professional[lang]
        detail[f"personal_skills_{lang}"] = personal[lang]

    detail["professional_skills_chips"] = "" if professional["en"] else \
        ", ".join(skills_from_visible_text(soup, "Professional Skills"))
    detail["personal_skills_chips"] = "" if personal["en"] else \
        ", ".join(skills_from_visible_text(soup, "Personal skills"))

    level = (job.get("candidate_level") or {}).get("title") or {}
    detail["candidate_level"] = level.get("en", "")
    detail["scraped_at"] = datetime.now(timezone.utc).isoformat()
    return detail


def scrape_category(category_id: int, max_pages: int = 10,
                    sleep_listing: float = 1.0, sleep_detail: float = 1.5) -> pd.DataFrame:
    """Scrape every posting in one staff.am category, listing pages first, then details."""
    session = requests.Session()
    jobs = []

    for page in range(1, max_pages + 1):
        url = f"https://staff.am/jobs?category={category_id}&key_word=&sort_by=0"
        if page > 1:
            url += f"&page={page}"
        try:
            response = session.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"listing page {page} failed: {str(e)[:150]}")
            continue

        page_data, _ = _next_data(response.content)
        listings = [j for j in page_data.get("jobs", []) if j.get("itemType") != 0]
        print(f"listing page {page}: {len(listings)} jobs")
        if not listings:
            break

        for job in listings:
            category_code = (job.get("category") or {}).get("code", "")
            slug = (job.get("slug") or {}).get("en", "")
            company = job.get("companiesStruct") or {}
            jobs.append({
                "job_id": job.get("id"),
                "title_en": (job.get("title") or {}).get("en", ""),
                "title_am": (job.get("title") or {}).get("am", ""),
                "title_ru": (job.get("title") or {}).get("ru", ""),
                "company_name": (company.get("title") or {}).get("en", ""),
                "category": ((job.get("category") or {}).get("title") or {}).get("en", ""),
                "city": ((job.get("job_city") or {}).get("title") or {}).get("en", ""),
                "deadline": job.get("deadline", ""),
                "is_remote": job.get("is_remote", False),
                "job_url": f"https://staff.am/en/jobs/{category_code}/{slug}",
            })
        time.sleep(sleep_listing)

    print(f"collected {len(jobs)} listings, fetching details...")
    for i, job in enumerate(jobs, 1):
        detail = get_job_detail(job["job_url"], session)
        if detail:
            job.update(detail)
        if i % 10 == 0 or i == len(jobs):
            print(f"  {i}/{len(jobs)}")
        time.sleep(sleep_detail)

    return pd.DataFrame(jobs)
