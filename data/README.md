# Data

## What's in this repo

| path | contents |
|---|---|
| `../results/skill_coverage.csv` | The final 165-skill table: demand, match step, similarity score, gap type and final status. Notebooks 02 and 03 run from this file alone. |
| `../results/threshold_sensitivity.csv` | Coverage at SBERT thresholds from 0.50 to 0.80. |

## What isn't, and why

* **The job-posting corpus (1,610 postings), including any sample of it.** The postings were collected
  from staff.am and job.am for research use and are not mine to redistribute.
  The scrapers in `src/skillgap/scrapers/` can rebuild a fresh corpus; listings
  change daily, so a new scrape will differ from the 2026 snapshot.
* **The AUA syllabi, and anything derived from them.** They're internal course
  documents, so the public results contain no syllabus text, topics, course
  names or instructor details: only skill-level results.

## Running the full pipeline

Put your own inputs in `data/private/` (git-ignored):

| file | columns |
|---|---|
| `top_canonical_skills.csv` | `skill, jobs, pct, domain` (domain: `CSE`, `BUS`, or `CSE+BUS`) |
| `syllabi_topics_canonicalized.csv` | `course_code, course_title, source_dataset, canonical_topics_pipe_v2` |
| `reviewed_gap_types.csv` | `skill, final_domain, gap_type` |
| `BUS_Syllabi.csv`, `CSE_Syllabi.csv` | one row per syllabus: `course_code, course_title, course_description, topics, course_learning_outcomes` (only needed for `scripts/extract_topics.py`) |
