"""Hybrid lexical + semantic matching of job-market skills to syllabus topics.

For every canonical skill the matcher tries, in order:

1. a lexical match against topics taught in the skill's own department,
2. the same lexical match against the other department (cross-department fallback),
3. Sentence-BERT cosine similarity against the department's topics.

A lexical match means every meaningful token of the skill appears in a topic
("Cloud Computing" needs both "cloud" and "computing"). It is transparent and
easy to audit, so it always wins when it fires. SBERT only decides the skills
that lexical matching can't resolve, and a skill counts as covered when its
best topic scores at or above the threshold (0.60 in the thesis).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SBERT_MODEL = "all-MiniLM-L12-v2"
SBERT_THRESHOLD = 0.60

# Words that carry no signal when comparing a skill name with a topic name.
STOP_WORDS = {
    "and", "the", "of", "for", "in", "a", "an", "to", "with", "or", "is", "are",
    "&", "at", "by", "on", "as", "its", "their", "this", "that", "from", "into",
    "using", "based", "via", "over", "under", "between", "within", "across",
}

DEPARTMENTS = ("CSE", "BUS")


def split_pipe(value) -> list[str]:
    """Parse a pipe-separated string ("A | B | C") into a clean list."""
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def to_tokens(text: str) -> set[str]:
    """Break a phrase into a set of meaningful lowercase tokens.

    Splits on whitespace and common separators, then drops stop words.

    >>> sorted(to_tokens("CI/CD"))
    ['cd', 'ci']
    >>> sorted(to_tokens("Node.js"))
    ['js', 'node']
    """
    parts = re.split(r"[\s/\.\-\(\)\,\:]+", text.lower())
    return {p for p in parts if p and p not in STOP_WORDS}


def own_and_other_departments(domain: str) -> tuple[list[str], list[str]]:
    """Departments to search first, and the fallback department (if any).

    Domain labels follow the syllabus data: "CSE", "BUS", or "CSE+BUS" for the
    few skills that were in demand in both areas.
    """
    if domain == "CSE":
        return ["CSE"], ["BUS"]
    if domain == "BUS":
        return ["BUS"], ["CSE"]
    return ["CSE", "BUS"], []


@dataclass
class Course:
    code: str
    title: str
    dept: str
    topics: set[str]
    topic_tokens: dict[str, set[str]] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.code} | {self.title}"


def build_course_index(syllabi: pd.DataFrame, topics_col: str = "canonical_topics_pipe_v2") -> dict[str, Course]:
    """Index courses by code, with the canonical topics each course teaches.

    Two syllabi in the AUA data have no course code. Keying on a blank code
    silently overwrites the first course with the second (and loses topics such
    as Power BI), so we fall back to the course title as the key.
    """
    courses: dict[str, Course] = {}
    for _, row in syllabi.iterrows():
        code = str(row["course_code"]).strip()
        title = str(row["course_title"]).strip()
        if code == "nan" or not code:
            code = f"NO_CODE_{title[:40]}"
        topics = split_pipe(row[topics_col])
        courses[code] = Course(
            code=code,
            title=title,
            dept=row["source_dataset"],
            topics=set(topics),
            topic_tokens={t: to_tokens(t) for t in topics},
        )
    return courses


@dataclass
class MatchResult:
    skill: str
    domain: str
    step: int            # 1 = own-dept lexical, 2 = cross-dept lexical, 3 = SBERT
    status: str          # "COVERED" or "MISSING"
    score: float         # 1.0 for lexical matches, cosine similarity for SBERT
    best_match: str
    covered_by: list[str]
    coverage_strength: str


class HybridMatcher:
    """Match skills to syllabus topics: lexical first, SBERT as a fallback."""

    def __init__(self, courses: dict[str, Course], model=None,
                 threshold: float = SBERT_THRESHOLD, model_name: str = SBERT_MODEL):
        self.courses = courses
        self.threshold = threshold
        self.topics = sorted({t for c in courses.values() for t in c.topics})
        self._topic_index = {t: i for i, t in enumerate(self.topics)}
        self._model = model
        self._model_name = model_name
        self._topic_vecs = None

    # -- embeddings -------------------------------------------------------
    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def similarity_matrix(self, skills: list[str]) -> np.ndarray:
        """Cosine similarity between every skill and every topic (skills x topics)."""
        from sklearn.metrics.pairwise import cosine_similarity

        if self._topic_vecs is None:
            self._topic_vecs = self.model.encode(self.topics, convert_to_numpy=True)
        skill_vecs = self.model.encode(skills, convert_to_numpy=True)
        return cosine_similarity(skill_vecs, self._topic_vecs)

    # -- the two search strategies ---------------------------------------
    def keyword_search(self, skill_tokens: set[str], depts: list[str]) -> tuple[str, list[str]]:
        """First topic (in course order) whose tokens contain all skill tokens."""
        best_topic, matched = "", []
        for course in self.courses.values():
            if course.dept not in depts:
                continue
            for topic, topic_tokens in course.topic_tokens.items():
                if skill_tokens and skill_tokens.issubset(topic_tokens):
                    best_topic = best_topic or topic
                    matched.append(course.label)
        return best_topic, list(dict.fromkeys(matched))

    def sbert_search(self, sims: np.ndarray, domain: str) -> tuple[str, float, str, list[str]]:
        """Best-scoring topic among the domain's departments, given one row of the similarity matrix."""
        depts, _ = own_and_other_departments(domain)
        candidates = sorted({t for c in self.courses.values() if c.dept in depts for t in c.topics})
        indices = [self._topic_index[t] for t in candidates] or list(range(len(self.topics)))

        best_i = max(indices, key=lambda i: sims[i])
        best_score = float(sims[best_i])
        best_topic = self.topics[best_i]

        if best_score >= self.threshold:
            covered_by = [c.label for c in self.courses.values()
                          if best_topic in c.topics and c.dept in depts]
            return "COVERED", round(best_score, 4), best_topic, list(dict.fromkeys(covered_by))
        return "MISSING", round(best_score, 4), best_topic, []

    # -- full pipeline ----------------------------------------------------
    def match(self, skills: pd.DataFrame, skill_col: str = "skill", domain_col: str = "domain") -> list[MatchResult]:
        names = skills[skill_col].astype(str).tolist()
        sim = self.similarity_matrix(names)

        results = []
        for i, (skill, domain) in enumerate(zip(names, skills[domain_col].astype(str))):
            tokens = to_tokens(skill)
            own, other = own_and_other_departments(domain)

            topic, courses = self.keyword_search(tokens, own)
            if topic:
                results.append(MatchResult(skill, domain, 1, "COVERED", 1.0, topic,
                                           courses, "Lexical match"))
                continue

            if other:
                topic, courses = self.keyword_search(tokens, other)
                if topic:
                    results.append(MatchResult(skill, domain, 2, "COVERED", 1.0, topic,
                                               courses, "Cross-department lexical match"))
                    continue

            status, score, topic, courses = self.sbert_search(sim[i], domain)
            strength = ("Semantic match above threshold" if status == "COVERED"
                        else "Below semantic threshold")
            results.append(MatchResult(skill, domain, 3, status, score, topic, courses, strength))
        return results


def results_to_frame(results: list[MatchResult], skills: pd.DataFrame) -> pd.DataFrame:
    """Combine match results with the demand columns (jobs, pct) of the skill table."""
    out = pd.DataFrame([{
        "skill": r.skill,
        "domain": r.domain,
        "step": r.step,
        "status": r.status,
        "score": r.score,
        "best_match": r.best_match,
        "coverage_strength": r.coverage_strength,
        "covered_by": "; ".join(r.covered_by[:3]),
    } for r in results])
    demand = skills.rename(columns=str.lower)[["skill", "jobs", "pct"]]
    return demand.merge(out, on="skill", how="right")[
        ["skill", "jobs", "pct", "domain", "step", "status", "score",
         "best_match", "coverage_strength", "covered_by"]
    ]
