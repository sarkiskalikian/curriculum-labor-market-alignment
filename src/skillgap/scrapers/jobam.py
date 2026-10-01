"""Scraper for job.am listings.

job.am renders plain HTML, and many postings are written in Armenian or
Russian with section headers in either language. The detail parser finds the
description / responsibilities / requirements / benefits sections by matching
their headers in all three languages, keeps bullet points as "- item", and then
routes each text block into a *_en / *_am / *_ru column based on its script, so
the output lines up with the staff.am columns.

Category IDs used in the study: 17 = IT / software development,
16 = Human resources, 39 = Banking & finance (plus others listed in the README).
"""

from __future__ import annotations

import re
import time
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup, NavigableString, Tag

from ..text import detect_language

BASE_URL = "https://job.am"

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.8,hy;q=0.5,ru;q=0.4",
    "Connection": "keep-alive",
}

SECTION_KEYWORDS = {
    "description": ["description", "job description",
                    "աշխատանքի նկարագրություն", "նկարագրություն", "описание"],
    "responsibilities": ["responsibilities", "duties",
                         "պարտականություններ", "աշխատանքային պարտականություններ", "обязанности"],
    "requirements": ["requirements", "qualifications", "skills",
                     "պահանջներ", "անհրաժեշտ հմտություններ", "պահանջվող հմտություններ",
                     "требования", "квалификация", "навыки"],
    "benefits": ["benefits", "offer", "what we offer",
                 "մենք առաջարկում ենք", "առաջարկվող հնարավորություններ",
                 "условия", "мы предлагаем"],
}

META_LABELS = {
    "location_detail": ["location", "քաղաք", "տեղանք", "տեղակայություն", "место", "город"],
    "salary_detail": ["salary", "աշխատավարձ", "վարձատրություն", "зарплата", "оплата"],
    "employment_type": ["employment type", "զբաղվածության", "тип занятости"],
    "work_schedule": ["work schedule", "աշխատաժամանակ", "գրաֆիկ", "график"],
    "work_experience": ["work experience", "աշխատանքային փորձ", "опыт работы"],
    "industry": ["industry", "ոլորտ", "сфера", "отрасль"],
    "deadline_detail": ["deadline", "վերջնաժամկետ", "крайний срок", "срок подачи"],
}


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _match(text: str, table: dict[str, list[str]]) -> str | None:
    t = text.lower().strip()
    for key, keywords in table.items():
        if any(kw in t for kw in keywords):
            return key
    return None


def fetch(session: requests.Session, url: str, retries: int = 3) -> str:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            r.encoding = "utf-8"
            return r.text
        except requests.RequestException as e:
            last_err = e
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"failed to fetch {url}: {last_err}")


def parse_listing_page(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    cards = soup.find_all("div", class_="jobs-card d-md-block d-none") or soup.find_all("div", class_="jobs-card")

    jobs = []
    for card in cards:
        link = card.find("a", class_="jttl") or card.find("a", class_="text-dark")
        if not link or not link.get("href"):
            continue

        company = card.find("span", class_="company")
        company_name = ""
        if company:
            company_name = _clean((company.find("a") or company).get_text(" ", strip=True))

        def tooltip(name):
            span = card.find("span", {"data-original-title": name})
            if not span:
                return ""
            inner = span.find("span", class_="va-middle") or span
            return _clean(inner.get_text(" ", strip=True))

        jobs.append({
            "job_title_raw": _clean(link.get_text(" ", strip=True)),
            "job_url": urljoin(BASE_URL, link["href"]),
            "company_name": company_name,
            "location": tooltip("City"),
            "deadline": tooltip("Deadline"),
            "salary": tooltip("Salary"),
        })
    return jobs


def _text_until_next_header(header: Tag, container: Tag) -> str:
    lines = []
    for el in header.next_elements:
        if el is header:
            continue
        if isinstance(el, Tag) and (container not in el.parents or el.name in {"h2", "h3", "h4"}):
            break
        if isinstance(el, Tag) and el.name == "li":
            t = _clean(el.get_text(" ", strip=True))
            if t:
                lines.append("- " + t)
        elif isinstance(el, Tag) and el.name == "p":
            t = _clean(el.get_text(" ", strip=True))
            if t:
                lines.append(t)
        elif isinstance(el, NavigableString):
            t = _clean(str(el))
            if len(t) > 3:
                lines.append(t)

    deduped = [x for i, x in enumerate(lines) if i == 0 or lines[i - 1] != x]
    return "\n".join(deduped).strip()


def parse_job_detail(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    details = {key: "" for key in META_LABELS}
    details.update({"description_raw": "", "responsibilities_raw": "", "requirements_raw": "",
                    "benefits": "", "company_website": "", "company_description": ""})

    for div in soup.select("div.info-address"):
        full = _clean(div.get_text(" ", strip=True))
        value_el = div.find("span", class_=re.compile(r"(txt-bold|font-weight-bold)"))
        value = _clean(value_el.get_text(" ", strip=True)) if value_el else ""
        key = _match(full.replace(value, "").strip(" :"), META_LABELS)
        if key and value:
            details[key] = value

    body = soup.find("div", class_="job-descr") or soup.find("div", class_="work-description")
    if body:
        for header in body.find_all(["h2", "h3", "h4"]):
            section = _match(_clean(header.get_text(" ", strip=True)), SECTION_KEYWORDS)
            content = _text_until_next_header(header, body) if section else ""
            if content:
                details["benefits" if section == "benefits" else f"{section}_raw"] = content
        if not details["description_raw"]:
            details["description_raw"] = _clean(body.get_text("\n", strip=True))

    about = soup.find("div", class_="company-info") or soup.find("div", class_="about-company")
    if about:
        link = about.find("a", href=re.compile(r"^http"))
        if link:
            details["company_website"] = link.get("href", "").strip()
        desc = about.find("div", class_="company-descr")
        if desc:
            details["company_description"] = _clean(desc.get_text(" ", strip=True))
    return details


def to_language_columns(job: dict) -> dict:
    """Route title / description / responsibilities / requirements into *_en/_am/_ru columns."""
    row = {
        "company_name": job.get("company_name", ""),
        "job_url": job.get("job_url", ""),
        "location": job.get("location") or job.get("location_detail", ""),
        "salary": job.get("salary") or job.get("salary_detail", ""),
        "deadline": job.get("deadline") or job.get("deadline_detail", ""),
        "employment_type": job.get("employment_type", ""),
        "work_schedule": job.get("work_schedule", ""),
        "work_experience": job.get("work_experience", ""),
        "industry": job.get("industry", ""),
        "benefits": job.get("benefits", ""),
        "company_website": job.get("company_website", ""),
        "company_description": job.get("company_description", ""),
    }
    for source, target in [("job_title_raw", "title"), ("description_raw", "full_description"),
                           ("responsibilities_raw", "responsibilities"), ("requirements_raw", "requirements")]:
        text = job.get(source, "")
        lang = detect_language(text)
        lang = lang if lang in {"am", "ru"} else "en"
        for code in ("en", "am", "ru"):
            row[f"{target}_{code}"] = text if code == lang else ""
    row["detected_language"] = detect_language(job.get("job_title_raw", "") + " " + job.get("description_raw", ""))
    return row


def scrape_category(category_id: int, keyword: str = "", max_pages: int = 12,
                    sleep_listing: float = 1.2, sleep_detail: float = 1.8) -> pd.DataFrame:
    """Scrape every posting in one job.am category (listing pages, then each job page)."""
    session = requests.Session()
    jobs, prev_urls, repeats = [], set(), 0

    for page in range(1, max_pages + 1):
        url = f"{BASE_URL}/en/jobs?i={category_id}&q={keyword}&p={page}"
        try:
            listing = parse_listing_page(fetch(session, url))
        except RuntimeError as e:
            print(e)
            continue
        if not listing:
            break

        # job.am keeps serving the last page when you ask for one past the end
        urls = {j["job_url"] for j in listing}
        repeats = repeats + 1 if urls == prev_urls else 0
        if repeats >= 2:
            break
        prev_urls = urls

        print(f"listing page {page}: {len(listing)} jobs")
        jobs.extend(listing)
        time.sleep(sleep_listing)

    unique = list({j["job_url"]: j for j in jobs}.values())
    print(f"collected {len(unique)} unique listings, fetching details...")

    for i, job in enumerate(unique, 1):
        try:
            job.update(parse_job_detail(fetch(session, job["job_url"])))
        except RuntimeError as e:
            print(f"  detail failed: {e}")
        if i % 10 == 0 or i == len(unique):
            print(f"  {i}/{len(unique)}")
        time.sleep(sleep_detail)

    return pd.DataFrame([to_language_columns(j) for j in unique])
