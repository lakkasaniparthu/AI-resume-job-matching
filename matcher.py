"""Starter TF-IDF baseline for ranking fictional job descriptions by resume similarity."""

import argparse
import csv
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_jobs(csv_path: Path):
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {"job_title", "description"}.issubset(reader.fieldnames):
            raise ValueError("CSV must have 'job_title' and 'description' columns.")
        jobs = [row for row in reader if (row.get("description") or "").strip()]
    if not jobs:
        raise ValueError("No job descriptions found in the CSV.")
    return jobs


def rank_jobs(resume: str, jobs: list[dict]):
    documents = [resume] + [job["description"] for job in jobs]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    vectors = vectorizer.fit_transform(documents)
    scores = cosine_similarity(vectors[0:1], vectors[1:]).ravel()
    return sorted(zip(jobs, scores), key=lambda item: item[1], reverse=True)


def main():
    parser = argparse.ArgumentParser(description="Rank jobs by TF-IDF cosine similarity.")
    parser.add_argument("--resume", type=Path, default=Path("sample_resume.txt"))
    parser.add_argument("--jobs", type=Path, default=Path("sample_jobs.csv"))
    args = parser.parse_args()

    try:
        resume = args.resume.read_text(encoding="utf-8").strip()
        if not resume:
            raise ValueError("Resume file is empty.")
        jobs = load_jobs(args.jobs)
        ranked = rank_jobs(resume, jobs)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")

    print("Job rankings (TF-IDF cosine similarity; higher means more similar text):")
    for i, (job, score) in enumerate(ranked, start=1):
        print(f"{i}. {job['job_title']}: {score:.3f}")


if __name__ == "__main__":
    main()
