"""
Inspect AI task for research waste screening using dynamic schema extraction.

This module provides tasks for:
1. Extracting screening criteria schema from a cheatsheet PDF
2. Screening PubMed articles using the extracted schema
3. Categorizing results into included/excluded RIS files
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import datamodel_code_generator
import rispy
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from pymed import PubMed

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatMessageUser,
    ContentDocument,
    ContentText,
    GenerateConfig,
    ResponseSchema,
)
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState, generate
from inspect_ai.util import json_schema

# ruff: noqa: F841, E402

load_dotenv()

# ============================================================================
# Configuration
# ============================================================================

CHEATSHEET_DEFAULT_PDF = Path(os.getenv("CHEATSHEET_DEFAULT_PDF"))
CHEATSHEET_PROVIDER = "google/gemini-2.5-pro-latest"
CHEATSHEET_SYSTEM_INSTRUCTION = (
    "You extract accurately data from a systematic review screening cheatsheet."
)

OUTPUT_DIR = Path.cwd() / "outputs"
SCHEMAS_DIR = OUTPUT_DIR / "schemas"

# ============================================================================
# Schema Models
# ============================================================================

from cheatsheet_parser_agent import CheatsheetParser

MetaSchema = CheatsheetParser.MetaSchema


def find_latest_metaschema(dir_path: Path | str = SCHEMAS_DIR) -> Path:
    """Non-recursive search for `metaschema_*.json` within `dir_path`."""
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise ValueError(f"{dir_path} is not a valid directory")

    # Regex to extract timestamp from filename
    pattern = re.compile(r"metaschema_(\d{8}_\d{6})\.json")

    metaschema_files = []
    for f in dir_path.glob("metaschema_*.json"):
        match = pattern.match(f.name)
        if match:
            timestamp = match.group(1)
            metaschema_files.append((timestamp, f))

    if not metaschema_files:
        raise FileNotFoundError("No metaschema JSON files found in directory")

    # Pick the file with the latest timestamp
    latest_file: Path = max(metaschema_files, key=lambda x: x[0])[1]

    return latest_file


def read_latest_metaschema(
    dir_path: Path | str = SCHEMAS_DIR, pydantic_model_name: str = MetaSchema.__name__
) -> MetaSchema:
    """Returns a validated `MetaSchema` instance."""
    latest_file = find_latest_metaschema(dir_path)
    with open(latest_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return MetaSchema.model_validate(data)


def build_extraction_schema_pydantic_path(metaschema_file: Path) -> Path:
    """Constructs expected path to extraction schema's Pydantic; does not check if it exists."""
    return metaschema_file.with_name(
        metaschema_file.stem.replace("metaschema_", "extraction_schema_") + ".py"
    )


def construct_latest_extraction_schema_pydantic_path(
    dir_path: Path | str = SCHEMAS_DIR,
) -> Path:
    """Finds latest `MetaSchema` and constructs Pydantic path; does not check if it exists."""
    latest_metaschema_file = find_latest_metaschema(dir_path)
    return build_extraction_schema_pydantic_path(latest_metaschema_file)


def dump_latest_extraction_schema_pydantic(dir_path: Path | str = SCHEMAS_DIR) -> Path:
    """Generates Pydantic class code for latest `MetaSchema`'s extraction schema and returns Path to it."""
    latest_metaschema_file = find_latest_metaschema(dir_path)
    with open(latest_metaschema_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    metaschema_instance = MetaSchema.model_validate(data)
    extraction_schema = metaschema_instance.build_extraction_schema()
    pydantic_class_dump_path = build_extraction_schema_pydantic_path(
        latest_metaschema_file
    )
    datamodel_code_generator.generate(
        input_=json.dumps(extraction_schema.model_json_schema()),
        input_file_type=datamodel_code_generator.InputFileType.JsonSchema,
        output_model_type=datamodel_code_generator.DataModelType.PydanticV2BaseModel,
        output=pydantic_class_dump_path,
    )
    return pydantic_class_dump_path


# ============================================================================
# Article Fetching
# ============================================================================


class Article(BaseModel):
    """Represents a PubMed article."""

    pmid: str
    title: str
    abstract: str
    authors: List[Dict[str, str | None]] = Field(default_factory=list)
    journal: str
    publication_date: Optional[datetime] = None
    doi: str = ""


def fetch_pubmed_articles(pmids: List[str | int]) -> List[Article]:
    """Fetch articles from PubMed API."""
    email = os.getenv("PUBMED_EMAIL", "inspect_task@example.com")
    tool = os.getenv("PUBMED_TOOL_NAME", "inspect-screening-task")

    pubmed = PubMed(tool=tool, email=email)
    query = " OR ".join([f"{pmid}[pmid]" for pmid in pmids])
    results = pubmed.query(query, max_results=len(pmids))

    articles = []
    for article in results:
        pmid_raw = getattr(article, "pubmed_id", "")
        pmid = str(pmid_raw).split("\n")[0] if pmid_raw else ""

        articles.append(
            Article(
                pmid=pmid,
                title=getattr(article, "title", "N/A"),
                abstract=getattr(article, "abstract", "N/A"),
                authors=getattr(article, "authors", []),
                journal=getattr(article, "journal", ""),
                publication_date=getattr(article, "publication_date", None),
                doi=str(getattr(article, "doi", "")),
            )
        )

    return articles


def articles_to_ris(articles: List[Article]) -> List[Dict[str, Any]]:
    """Convert articles to RIS format."""
    ris_entries = []

    for article in articles:
        authors = []
        if article.authors:
            for author in article.authors:
                if isinstance(author, dict):
                    lastname = author.get("lastname", "")
                    firstname = author.get("firstname", "")
                    if lastname and firstname:
                        authors.append(f"{lastname} {firstname}".strip())

        entry = {
            "type_of_reference": "JOUR",
            "authors": authors,
            "year": (article.publication_date.year if article.publication_date else ""),
            "title": article.title,
            "journal_name": article.journal,
            "abstract": article.abstract,
            "ID": article.pmid,
            "doi": article.doi,
        }

        # Remove empty fields
        entry = {k: v for k, v in entry.items() if v}
        ris_entries.append(entry)

    return ris_entries


def save_ris_file(ris_entries: List[Dict[str, Any]], filepath: Path) -> None:
    """Save RIS entries to file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", encoding="utf-8-sig") as f:
        rispy.dump(ris_entries, f)


# ============================================================================
# Task 1: Schema Extraction from Cheatsheet
# ============================================================================


@task
def extract_cheatsheet_schema(
    cheatsheet_pdf: str = str(CHEATSHEET_DEFAULT_PDF),
    output_dir: str = "outputs/schemas",
):
    """
    Task to extract screening schema from a cheatsheet PDF.

    Args:
        cheatsheet_pdf: Path to cheatsheet PDF
        output_dir: Directory to save extracted schema
    """
    pdf_path = Path(cheatsheet_pdf).expanduser().resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(f"Cheatsheet PDF not found: {pdf_path}")

    prompt = """
You are given a systematic review screening cheatsheet. Extract its structured schema.

The cheatsheet is attached.

Analyze the document and produce a complete MetaSchema that captures:
1. All screening criteria fields
2. Field types (string, enum with values, list, etc)
3. Field descriptions
4. Which fields are required

Focus on extracting the decision logic and categorization criteria.
"""

    return Task(
        dataset=[
            Sample(
                input=[
                    ChatMessageUser(
                        content=[
                            ContentText(text=prompt),
                            ContentDocument(document=str(pdf_path)),
                        ]
                    )
                ],
                id="cheatsheet_schema_extraction",
            )
        ],
        solver=generate(),
        scorer=save_metaschema(output_dir=output_dir),
        config=GenerateConfig(
            response_schema=ResponseSchema(
                name=MetaSchema.__name__, json_schema=json_schema(MetaSchema)
            ),
            temperature=0.0,
        ),
    )


@scorer(metrics=[accuracy(), stderr()])
def save_metaschema(output_dir: str = "outputs/schemas"):
    """Scorer that saves the extracted metaschema to disk."""

    async def score(state: TaskState, target: Target) -> Score:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            # Parse the metaschema
            metaschema = MetaSchema.model_validate_json(state.output.completion)

            # Save as JSON
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            json_path = output_path / f"metaschema_{timestamp}.json"

            with json_path.open("w", encoding="utf-8") as f:
                f.write(metaschema.model_dump_json(indent=2))

            # Build and save extraction schema
            extraction_schema = metaschema.build_extraction_schema()
            schema_json_path = output_path / f"extraction_schema_{timestamp}.json"

            with schema_json_path.open("w", encoding="utf-8") as f:
                json.dump(
                    extraction_schema.model_json_schema(),
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            return Score(
                value=CORRECT,
                answer=state.output.completion,
                explanation=f"Metaschema saved to {json_path}",
                metadata={
                    "metaschema_path": str(json_path),
                    "schema_path": str(schema_json_path),
                },
            )

        except ValidationError as ex:
            return Score(
                value=INCORRECT,
                answer=state.output.completion,
                explanation=f"Failed to parse metaschema: {ex}",
            )

    return score


# ============================================================================
# Task 2: Article Screening
# ============================================================================


@task
def screen_articles(
    pmids: List[str],
    metaschema_json: str,
    output_dir: str = "outputs/screening",
):
    """
    Task to screen articles using extracted schema.

    Args:
        pmids: List of PubMed IDs to screen
        metaschema_json: Path to saved metaschema JSON
        output_dir: Directory for screening outputs
    """
    # Load metaschema
    schema_path = Path(metaschema_json).expanduser().resolve()
    with schema_path.open("r", encoding="utf-8") as f:
        metaschema_dict = json.load(f)

    metaschema = MetaSchema.model_validate(metaschema_dict)
    extraction_schema = metaschema.build_extraction_schema()

    # Fetch articles
    articles = fetch_pubmed_articles(pmids)

    # Create samples for each article
    samples = []
    pydantic_code = open(construct_latest_extraction_schema_pydantic_path()).read()
    for article in articles:
        prompt = f"""
Screen this article according to the systematic review criteria.

Title: {article.title}

Abstract: {article.abstract}

Provide your screening decision following the schema structure:

```python
{pydantic_code}
```
"""
        samples.append(
            Sample(
                input=prompt,
                target=json.dumps({"pmid": article.pmid}),
                id=article.pmid,
                metadata={
                    "pmid": article.pmid,
                    "title": article.title,
                    "abstract": article.abstract,
                },
            )
        )

    return Task(
        dataset=samples,
        solver=generate(),
        scorer=categorize_articles(
            articles=articles,
            output_dir=output_dir,
            extraction_schema=extraction_schema,
        ),
        config=GenerateConfig(
            response_schema=ResponseSchema(
                name=extraction_schema.__name__,
                json_schema=json_schema(extraction_schema),
            ),
            temperature=0.0,
            max_tokens=8192,
        ),
    )


@scorer(metrics=[accuracy(), stderr()])
def categorize_articles(
    articles: List[Article],
    output_dir: str,
    extraction_schema: type[BaseModel],
):
    """Scorer that categorizes articles and generates RIS files."""

    async def score(state: TaskState, target: Target) -> Score:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        pmid = state.metadata.get("pmid", "unknown")

        try:
            # Parse screening output
            screening_result = extraction_schema.model_validate_json(
                state.output.completion
            )

            # Extract decision (assumes schema has a decision field)
            # This is a simplification - adapt based on actual schema
            decision_field = None
            for field_name in type(screening_result).model_fields.keys():
                if (
                    "decision" in field_name.lower()
                    or "screening" in field_name.lower()
                ):
                    decision_field = field_name
                    break

            if decision_field:
                decision_value = getattr(screening_result, decision_field, None)
                decision_str = str(decision_value).lower()

                if "include" in decision_str:
                    category = "included"
                    score_value = CORRECT
                elif "exclude" in decision_str:
                    category = "excluded"
                    score_value = CORRECT
                else:
                    category = "undecided"
                    score_value = INCORRECT
            else:
                category = "undecided"
                score_value = INCORRECT

            # Save individual result
            result_path = output_path / f"result_{pmid}.json"
            with result_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "pmid": pmid,
                        "category": category,
                        "screening_output": screening_result.model_dump(),
                    },
                    f,
                    indent=2,
                )

            return Score(
                value=score_value,
                answer=state.output.completion,
                explanation=f"Article {pmid} categorized as {category}",
                metadata={
                    "pmid": pmid,
                    "category": category,
                    "result_path": str(result_path),
                },
            )

        except ValidationError as ex:
            # Save error result
            error_path = output_path / f"error_{pmid}.json"
            with error_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "pmid": pmid,
                        "category": "invalid_syntax",
                        "error": str(ex),
                        "raw_output": state.output.completion,
                    },
                    f,
                    indent=2,
                )

            return Score(
                value=INCORRECT,
                answer=state.output.completion,
                explanation=f"Failed to parse screening for {pmid}: {ex}",
                metadata={"pmid": pmid, "category": "invalid_syntax"},
            )

    return score


@task
def generate_ris_outputs(screening_output_dir: str, pmids: List[str]):
    """
    Post-processing task to generate categorized RIS files.

    Args:
        screening_output_dir: Directory containing screening results
        pmids: List of PMIDs that were screened
    """
    output_path = Path(screening_output_dir)

    # Fetch all articles
    articles = fetch_pubmed_articles(pmids)
    articles_by_pmid = {a.pmid: a for a in articles}

    # Categorize based on screening results
    included = []
    excluded = []
    undecided = []

    for pmid in pmids:
        result_file = output_path / f"result_{pmid}.json"
        error_file = output_path / f"error_{pmid}.json"

        article = articles_by_pmid.get(pmid)
        if not article:
            continue

        category = "undecided"

        if result_file.exists():
            with result_file.open("r", encoding="utf-8") as f:
                result = json.load(f)
                category = result.get("category", "undecided")
        elif error_file.exists():
            category = "undecided"

        if category == "included":
            included.append(article)
        elif category == "excluded":
            excluded.append(article)
        else:
            undecided.append(article)

    # Generate RIS files
    if included:
        save_ris_file(articles_to_ris(included), output_path / "included.ris")

    if excluded:
        save_ris_file(articles_to_ris(excluded), output_path / "excluded.ris")

    if undecided:
        save_ris_file(articles_to_ris(undecided), output_path / "undecided.ris")

    # Create summary sample
    summary = Sample(
        input="Generate RIS files from screening results",
        target=json.dumps(
            {
                "included": len(included),
                "excluded": len(excluded),
                "undecided": len(undecided),
            }
        ),
        metadata={
            "included_count": len(included),
            "excluded_count": len(excluded),
            "undecided_count": len(undecided),
            "output_dir": str(output_path),
        },
    )

    return Task(
        dataset=[summary],
        solver=[],  # No solver needed - post-processing only
        scorer=ris_generation_scorer(),
    )


@scorer(metrics=[accuracy()])
def ris_generation_scorer():
    """Simple scorer for RIS generation task."""

    async def score(state: TaskState, target: Target) -> Score:
        metadata = state.metadata
        return Score(
            value=CORRECT,
            explanation=(
                f"Generated RIS files: "
                f"{metadata['included_count']} included, "
                f"{metadata['excluded_count']} excluded, "
                f"{metadata['undecided_count']} undecided"
            ),
            metadata=metadata,
        )

    return score


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    """
    Example usage:

    # Step 1: Extract schema from cheatsheet
    inspect eval inspect_screening_task.py@extract_cheatsheet_schema \\
        --model google/gemini-2.5-pro-latest

    # Step 2: Screen articles using extracted schema
    inspect eval inspect_screening_task.py@screen_articles \\
        --model google/gemini-2.5-pro-latest \\
        -T pmids='["10618008","21592376"]' \\
        -T metaschema_json="outputs/schemas/metaschema_TIMESTAMP.json"

    # Step 3: Generate RIS files
    inspect eval inspect_screening_task.py@generate_ris_outputs \\
        -T screening_output_dir="outputs/screening" \\
        -T pmids='["10618008","21592376"]'
    """
    pass
