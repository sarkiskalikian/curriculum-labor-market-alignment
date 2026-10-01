"""Extract raw skills from every posting, then canonicalize them.

Needs OPENAI_API_KEY in the environment. Extraction uses gpt-4.1-mini and
canonicalization GPT-4o, as in the thesis.

    python scripts/extract_skills.py data/interim/jobs_corpus.csv --out data/interim/jobs_skills.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skillgap.llm import (CANONICALIZATION_MODEL, EXTRACTION_MODEL, canonicalize,  # noqa: E402
                          extract_skills, get_client, map_pipe)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus")
    parser.add_argument("--out", required=True)
    parser.add_argument("--extraction-model", default=EXTRACTION_MODEL)
    parser.add_argument("--canonicalization-model", default=CANONICALIZATION_MODEL)
    parser.add_argument("--limit", type=int, help="only process the first N postings (for a dry run)")
    args = parser.parse_args()

    client = get_client()
    df = pd.read_csv(args.corpus)
    if args.limit:
        df = df.head(args.limit)

    # 1. raw extraction, one call per posting
    extracted = [extract_skills(client, t, args.extraction_model) for t in tqdm(df["job_content_for_extraction"])]
    df["skills_pipe"] = [" | ".join(skills) for skills, _ in extracted]
    df["extraction_status"] = [status for _, status in extracted]

    # 2. canonicalization, one call per distinct raw phrase
    raw = sorted({s.strip() for row in df["skills_pipe"] for s in str(row).split("|") if s.strip()})
    mapping = {r: canonicalize(client, r, "skill", args.canonicalization_model) for r in tqdm(raw)}
    df["canonical_skills_pipe"] = df["skills_pipe"].apply(lambda p: " | ".join(map_pipe(p, mapping)))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    pd.DataFrame({"raw_skill": list(mapping), "canonical_skill": list(mapping.values())}).to_csv(
        Path(args.out).with_name("skill_canonical_map.csv"), index=False, encoding="utf-8-sig")
    print(f"{len(raw)} raw phrases -> {len(set(mapping.values()))} canonical skills")


if __name__ == "__main__":
    main()
