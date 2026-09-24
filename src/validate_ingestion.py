
from pathlib import Path
import json
import re
from collections import Counter


# ==================================================
# PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[1]

VECTORSTORE_DIR = BASE_DIR / "vectorstore"
CLEANED_TEXT_DIR = BASE_DIR / "data" / "cleaned_text"


METADATA_PATH = VECTORSTORE_DIR / "metadata.json"


# ==================================================
# SETTINGS
# ==================================================

REQUIRED_FIELDS = [
    "text",
    "source",
    "document_id",
    "page",
    "section",
    "parent_id",
    "chunk_id",
    "parent_text"
]


# ==================================================
# VALIDATION FUNCTIONS
# ==================================================

def load_metadata():

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH}"
        )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def validate_required_fields(metadata):

    errors = []

    for index, record in enumerate(metadata):

        for field in REQUIRED_FIELDS:

            if field not in record:
                errors.append(
                    f"Record {index}: missing {field}"
                )

    return errors


def validate_empty_text(metadata):

    errors = []

    for index, record in enumerate(metadata):

        if not record.get("text", "").strip():
            errors.append(
                f"Record {index}: empty child text"
            )

        if not record.get("parent_text", "").strip():
            errors.append(
                f"Record {index}: empty parent text"
            )

    return errors


def validate_hierarchy(metadata):

    errors = []

    parent_to_children = {}

    for index, record in enumerate(metadata):

        parent_id = record["parent_id"]
        chunk_id = record["chunk_id"]

        parent_to_children.setdefault(
            parent_id,
            []
        ).append(chunk_id)

        child_text = record["text"]
        parent_text = record["parent_text"]

        if child_text not in parent_text:
            errors.append(
                f"Record {index}: child text not found "
                f"inside parent text"
            )

    for parent_id, child_ids in parent_to_children.items():

        if len(child_ids) == 0:
            errors.append(
                f"Parent {parent_id} has no children"
            )

    return errors


def validate_duplicate_chunk_ids(metadata):

    errors = []

    chunk_ids = [
        record["chunk_id"]
        for record in metadata
    ]

    counts = Counter(chunk_ids)

    duplicates = [
        chunk_id
        for chunk_id, count in counts.items()
        if count > 1
    ]

    if duplicates:

        errors.append(
            f"Duplicate chunk IDs: {duplicates}"
        )

    return errors


def inspect_cleaned_text():

    print("\nCLEANED TEXT INSPECTION")

    if not CLEANED_TEXT_DIR.exists():
        print("Cleaned text directory not found.")
        return

    text_files = sorted(
        CLEANED_TEXT_DIR.glob("*_cleaned.txt")
    )

    print(
        f"Cleaned text files found: {len(text_files)}"
    )

    noise_patterns = [
        r"\b[A-Za-z]{2,}-\s+[a-z]{2,}\b",
        r"\b[A-Z]\s+[a-z]{3,}\b"
    ]

    for text_file in text_files:

        text = text_file.read_text(
            encoding="utf-8"
        )

        print(f"\nFile: {text_file.name}")
        print(f"Characters: {len(text)}")
        print(f"Words: {len(text.split())}")

        for pattern in noise_patterns:

            matches = re.findall(
                pattern,
                text
            )

            if matches:

                print(
                    f"Possible extraction noise "
                    f"({pattern}): {matches[:10]}"
                )


# ==================================================
# MAIN VALIDATION
# ==================================================

if __name__ == "__main__":

    metadata = load_metadata()

    print("\n" + "=" * 60)
    print("INGESTION VALIDATION")
    print("=" * 60)

    print(
        f"Total metadata records: {len(metadata)}"
    )

    errors = []

    errors.extend(
        validate_required_fields(metadata)
    )

    errors.extend(
        validate_empty_text(metadata)
    )

    errors.extend(
        validate_hierarchy(metadata)
    )

    errors.extend(
        validate_duplicate_chunk_ids(metadata)
    )

    if errors:

        print("\nVALIDATION ERRORS:")

        for error in errors[:50]:
            print(f"- {error}")

        if len(errors) > 50:
            print(
                f"... and {len(errors) - 50} more errors"
            )

    else:

        print(
            "\nAll metadata and hierarchy checks passed."
        )

    inspect_cleaned_text()

    print("\nValidation completed.")