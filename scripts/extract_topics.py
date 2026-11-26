"""Turn the structured syllabus tables into canonical topics per course.

1. normalize: keep description / topics / learning outcomes, drop admin lines, one text per course
2. extract:   8-15 primary topics per course (gpt-4.1-mini)
3. canonicalize: map every raw topic to a short standard label (GPT-4o)

Needs OPENAI_API_KEY in the environment.

    python scripts/extract_topics.py data/private/BUS_Syllabi.csv data/private/CSE_Syllabi.csv \
        --out data/private/syllabi_topics_canonicalized.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skillgap.corpus import read_csv_any_encoding  # noqa: E402
from skillgap.llm import (CANONICALIZATION_MODEL, EXTRACTION_MODEL, canonicalize,  # noqa: E402
                          extract_topics, get_client, map_pipe)
from skillgap.syllabi import normalize_syllabi  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bus_syllabi")
    parser.add_argument("cse_syllabi")
    parser.add_argument("--out", required=True)
    parser.add_argument("--extraction-model", default=EXTRACTION_MODEL)
    parser.add_argument("--canonicalization-model", default=CANONICALIZATION_MODEL)
    args = parser.parse_args()

    _, courses = normalize_syllabi(read_csv_any_encoding(args.bus_syllabi), read_csv_any_encoding(args.cse_syllabi))
    print(f"{len(courses)} courses after normalization")

    client = get_client()
    extracted = [extract_topics(client, r.course_code, r.course_title, r.course_content_for_extraction,
                                args.extraction_model) for r in tqdm(courses.itertuples(), total=len(courses))]
    courses["primary_topics_pipe_v2"] = [" | ".join(topics) for topics, _ in extracted]
    courses["extraction_status"] = [status for _, status in extracted]

    raw = sorted({t.strip() for row in courses["primary_topics_pipe_v2"] for t in row.split("|") if t.strip()})
    mapping = {t: canonicalize(client, t, "topic", args.canonicalization_model) for t in tqdm(raw)}
    courses["canonical_topics_pipe_v2"] = courses["primary_topics_pipe_v2"].apply(lambda p: " | ".join(map_pipe(p, mapping)))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    courses.to_csv(args.out, index=False, encoding="utf-8-sig")
    n_topics = len({t for row in courses["canonical_topics_pipe_v2"] for t in row.split(" | ") if t})
    print(f"{len(raw)} raw topics -> {n_topics} canonical topics; saved to {args.out}")


if __name__ == "__main__":
    main()
