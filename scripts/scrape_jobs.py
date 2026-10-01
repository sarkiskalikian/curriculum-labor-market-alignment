"""Scrape one job-board category to CSV.

    python scripts/scrape_jobs.py staffam --category 1  --out data/raw/staffam_software.csv
    python scripts/scrape_jobs.py jobam   --category 17 --out data/raw/jobam_it_software_development.csv

Please keep the default delays; both sites are small and the scraper is meant
to be polite, not fast.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skillgap.scrapers import jobam, staffam  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("platform", choices=["staffam", "jobam"])
    parser.add_argument("--category", type=int, required=True, help="category id on the job board")
    parser.add_argument("--max-pages", type=int, default=10)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    module = staffam if args.platform == "staffam" else jobam
    df = module.scrape_category(args.category, max_pages=args.max_pages)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"saved {len(df)} postings to {out}")


if __name__ == "__main__":
    main()
