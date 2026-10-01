"""Merge the translated per-category CSVs into one de-duplicated corpus.

    python scripts/build_corpus.py data/raw/*__translated_EN.csv --out data/interim/jobs_corpus.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skillgap.corpus import build_corpus  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="translated CSV exports from the scrapers")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    corpus = build_corpus(args.inputs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    corpus.to_csv(args.out, index=False, encoding="utf-8-sig")

    print(f"{len(corpus)} postings kept")
    print(corpus["source_platform"].value_counts().to_string())


if __name__ == "__main__":
    main()
