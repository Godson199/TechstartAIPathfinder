# Add TechieStart source documents here

## Knowledge documents

Add the official source documents used for factual chatbot answers in this
folder:

- `participant_handbook.md` for the participant handbook
- `curriculum.md` for curriculum and programme details
- `faq.md` for Markdown FAQs
- `TechieStart_50_FAQs_Guide.csv` for FAQ rows with `Question`, `Answer`, and
  optional `Category` columns

Run the ingestion script after adding or updating documents:

```powershell
python scripts/ingest_knowledge.py
```

For Word or PDF documents, convert them to Markdown first. Do not add
confidential documents to source control.

## Pathfinder programme catalogue

Recommendations require a separate file named `programs.json` in this same
folder. Copy `programs.example.json` to `programs.json` and replace every
placeholder with authoritative programme information. This file is where you
add programme names, descriptions, prerequisites, skills, outcomes, career
directions, and durations used by the deterministic matcher.

The matcher returns no recommendation until this file contains real entries.
It will not infer programme names or details from the FAQ.

You can store the catalogue elsewhere by setting:

```powershell
$env:PATHFINDER_PROGRAM_CATALOG = "C:\path\to\programs.json"
```