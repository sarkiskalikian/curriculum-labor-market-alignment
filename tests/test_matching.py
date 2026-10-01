"""Unit tests for the lexical side of the matcher (no model download needed)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skillgap.matching import HybridMatcher, build_course_index, split_pipe, to_tokens  # noqa: E402
from skillgap.syllabi import clean_field  # noqa: E402
from skillgap.validation import apply_validation  # noqa: E402


@pytest.fixture
def courses():
    syllabi = pd.DataFrame({
        "course_code": ["TST 100", "TST 200", None],
        "course_title": ["Programming Basics", "Marketing Methods", "People Management"],
        "source_dataset": ["CSE", "BUS", "BUS"],
        "canonical_topics_pipe_v2": ["R for Beginners | Cloud Computing Basics", "Marketing Methods | Regression Lab",
                                     "Hiring | Power BI"],
    })
    return build_course_index(syllabi)


def test_tokens_split_on_separators_and_drop_stop_words():
    assert to_tokens("CI/CD") == {"ci", "cd"}
    assert to_tokens("Node.js") == {"node", "js"}
    assert to_tokens("Analysis of Variance") == {"analysis", "variance"}


def test_split_pipe_ignores_blanks():
    assert split_pipe("A | B |  | C") == ["A", "B", "C"]
    assert split_pipe(float("nan")) == []


def test_course_without_code_is_not_overwritten(courses):
    # the blank-code course must survive under a title-based key
    assert any(code.startswith("NO_CODE_") for code in courses)
    assert "Power BI" in {t for c in courses.values() for t in c.topics}


def test_keyword_match_needs_every_token(courses):
    matcher = HybridMatcher(courses, model=object())
    topic, _ = matcher.keyword_search(to_tokens("Cloud Computing"), ["CSE"])
    assert topic == "Cloud Computing Basics"
    topic, _ = matcher.keyword_search(to_tokens("Cloud Security"), ["CSE"])
    assert topic == ""


def test_single_letter_skill_matches_whole_word_only(courses):
    matcher = HybridMatcher(courses, model=object())
    topic, _ = matcher.keyword_search(to_tokens("R"), ["CSE", "BUS"])
    assert topic == "R for Beginners"        # not "Regression Lab"


def test_sbert_threshold_is_inclusive(courses):
    matcher = HybridMatcher(courses, model=object(), threshold=0.60)
    sims = np.zeros(len(matcher.topics))
    sims[matcher.topics.index("Marketing Methods")] = 0.60
    status, score, topic, covered_by = matcher.sbert_search(sims, "BUS")
    assert (status, topic) == ("COVERED", "Marketing Methods")
    assert covered_by == ["TST 200 | Marketing Methods"]


def test_validation_maps_gap_types_to_three_classes():
    df = pd.DataFrame({"gap_type": ["Lexical Coverage", "Semantic Coverage",
                                    "Partial / Documentation Gap", "Teaching Gap"]})
    assert apply_validation(df)["final_status"].tolist() == ["COVERED", "COVERED", "PARTIAL", "MISSING"]


def test_syllabus_cleaning_drops_admin_lines_and_duplicates():
    raw = "Week 1\nIntro to Databases\nMidterm\nSQL Joins; Normalization\nSQL Joins\nSpring break"
    assert clean_field(raw, "topics") == "Intro to Databases | SQL Joins | Normalization"


def test_syllabus_cleaning_reads_json_topic_lists():
    raw = '[{"week": 1, "topic": "Regression"}, {"week": 2, "topic": "Classification"}]'
    assert clean_field(raw, "topics") == "Regression | Classification"
