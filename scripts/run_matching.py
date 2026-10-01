"""Run the hybrid lexical + SBERT matcher.

    python scripts/run_matching.py \
        --skills data/private/top_canonical_skills.csv \
        --syllabi data/private/syllabi_topics_canonicalized.csv \
        --out data/private/matching_results.csv

The skills file needs columns: skill, jobs, pct, domain (CSE / BUS / CSE+BUS).
The syllabi file needs: course_code, course_title, source_dataset, canonical_topics_pipe_v2.
The output contains syllabus topics and course names, so keep it out of the repo.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skillgap.matching import SBERT_THRESHOLD, HybridMatcher, build_course_index, results_to_frame  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skills", required=True)
    parser.add_argument("--syllabi", required=True)
    parser.add_argument("--threshold", type=float, default=SBERT_THRESHOLD)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    skills = pd.read_csv(args.skills)
    syllabi = pd.read_csv(args.syllabi)
    syllabi = syllabi[syllabi["source_dataset"].isin(["CSE", "BUS"])]

    matcher = HybridMatcher(build_course_index(syllabi), threshold=args.threshold)
    results = results_to_frame(matcher.match(skills), skills)
    results.to_csv(args.out, index=False)

    covered = (results["status"] == "COVERED").sum()
    print(f"{covered}/{len(results)} skills covered ({covered / len(results):.1%}) at threshold {args.threshold}")
    print(results.groupby("step")["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
