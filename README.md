# AI Resume–Job Matching System

**Status: Early-stage prototype / work in progress**

This student project explores how Natural Language Processing (NLP) can help compare resume text with job descriptions. The long-term goal is to compare keyword matching, TF-IDF with cosine similarity, and semantic embeddings.

## Current implementation

This repository currently contains a **small proof-of-concept** that:
- Reads resume text from a local `.txt` file.
- Reads a small, **fictional demonstration dataset** of job descriptions from CSV.
- Converts the resume and job descriptions into TF-IDF vectors.
- Computes cosine similarity and prints jobs ranked by similarity.

The similarity score is **not** a hiring decision, match probability, or validated measure of candidate suitability. It is only a text-similarity baseline.

## Run locally

Requires Python 3.9 or newer.

```bash
python -m pip install -r requirements.txt
python matcher.py --resume sample_resume.txt --jobs sample_jobs.csv
```

Use your own plain-text resume if you prefer:

```bash
python matcher.py --resume path/to/my_resume.txt --jobs sample_jobs.csv
```

**Privacy note:** Do not commit real resumes, personal information, or private application data to a public repository. Sample files included here are fictional.

## Repository files

- `matcher.py` — TF-IDF and cosine-similarity starter script.
- `sample_jobs.csv` — synthetic job descriptions for demonstration only.
- `sample_resume.txt` — fictional sample resume.
- `requirements.txt` — Python dependency.
- `.gitignore` — ignores common local and private files.

## Planned work

1. Identify an appropriate real dataset and check its license and use conditions.
2. Develop and evaluate a simple keyword-matching baseline.
3. Improve preprocessing and compare against TF-IDF similarity.
4. Explore pre-trained sentence embeddings for semantic matching.
5. Define evaluation methods using labeled data, where available, and assess ranking quality.
6. Optionally create a simple user interface.

## Limitations

This is an initial implementation using fictional inputs. No formal experiments, accuracy results, real-world dataset evaluation, or validated semantic-matching model are included yet.
