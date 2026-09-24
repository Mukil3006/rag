
from pathlib import Path
import json
import re
from collections import Counter

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ==================================================
# PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
CLEANED_TEXT_DIR = BASE_DIR / "data" / "cleaned_text"
REPORTS_DIR = BASE_DIR / "data" / "reports"

VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
CLEANED_TEXT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# SETTINGS
# ==================================================

PARENT_CHUNK_SIZE = 450
CHILD_CHUNK_SIZE = 150
CHILD_OVERLAP = 25

EMBEDDING_MODEL = "BAAI/bge-m3"

# Lines appearing on multiple pages may be headers/footers.
# Keep this conservative to avoid deleting useful content.
REPEATED_LINE_MIN_PAGES = 2

# Maximum number of characters allowed for a probable heading.
MAX_HEADING_LENGTH = 150


# ==================================================
# TEXT NORMALIZATION
# ==================================================

def normalize_line(line):
    """
    Normalize a single extracted line for comparison.
    """

    line = line.replace("\u00a0", " ")
    line = re.sub(r"\s+", " ", line)

    return line.strip()


def fix_hyphenation(text):
    """
    Join words split across a line break.

    Example:
        retriev-
        al

    becomes:
        retrieval

    Only joins alphabetic words separated by a hyphen
    and a newline. Normal hyphenated words remain intact.
    """

    text = re.sub(
        r"([A-Za-z]{2,})-\s*\n\s*([a-z]{2,})",
        r"\1\2",
        text
    )

    return text


def normalize_whitespace(text):
    """
    Normalize spaces and line breaks while preserving
    paragraph boundaries where possible.
    """

    text = text.replace("\u00a0", " ")

    # Normalize Windows and old-style line endings.
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove spaces at the beginning/end of lines.
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_text(text):
    """
    Apply the basic text-cleaning pipeline.
    """

    text = fix_hyphenation(text)
    text = normalize_whitespace(text)

    return text.strip()


# ==================================================
# HEADING DETECTION
# ==================================================

def is_probable_heading(line):
    """
    Basic heuristic for detecting section headings.

    This is intentionally conservative. It does not
    guarantee that every detected line is a heading.
    """

    line = normalize_line(line)

    if not line:
        return False

    if len(line) > MAX_HEADING_LENGTH:
        return False

    words = line.split()

    if len(words) > 18:
        return False

    # Common numbered headings:
    # 1. Introduction
    # 2.1 Retrieval
    # 3 Methodology
    if re.match(
        r"^\d+(\.\d+)*[\.)]?\s+[A-Z]",
        line
    ):
        return True

    # Common heading labels.
    heading_keywords = (
        "abstract",
        "introduction",
        "background",
        "methodology",
        "methods",
        "results",
        "discussion",
        "conclusion",
        "references",
        "limitations",
        "evaluation",
        "architecture",
        "retrieval",
        "generation"
    )

    lower_line = line.lower()

    if lower_line in heading_keywords:
        return True

    # Short all-uppercase lines may be headings.
    if (
        len(words) <= 10
        and line.upper() == line
        and any(char.isalpha() for char in line)
    ):
        return True

    return False


def detect_section(line, current_section):
    """
    Update the current section if the line appears
    to be a section heading.
    """

    line = normalize_line(line)

    if is_probable_heading(line):
        return line

    return current_section


# ==================================================
# PAGE EXTRACTION
# ==================================================

def extract_pages(pdf_path):
    """
    Extract text from each PDF page while preserving
    page numbers and raw text.
    """

    reader = PdfReader(str(pdf_path))

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        raw_text = page.extract_text() or ""

        if not raw_text.strip():
            continue

        pages.append({
            "page": page_number,
            "raw_text": raw_text
        })

    return pages


# ==================================================
# REPEATED HEADER/FOOTER DETECTION
# ==================================================

def collect_repeated_lines(pages):
    """
    Identify normalized lines appearing on multiple
    pages.

    This is used as a conservative signal for removing
    repeated headers and footers.
    """

    line_pages = {}

    for page_data in pages:

        page_number = page_data["page"]

        raw_lines = page_data["raw_text"].splitlines()

        unique_lines = set()

        for line in raw_lines:

            normalized = normalize_line(line)

            if not normalized:
                continue

            unique_lines.add(normalized.lower())

        for line in unique_lines:

            if line not in line_pages:
                line_pages[line] = set()

            line_pages[line].add(page_number)

    repeated_lines = {
        line
        for line, page_numbers in line_pages.items()
        if len(page_numbers) >= REPEATED_LINE_MIN_PAGES
    }

    return repeated_lines


def is_page_number(line):
    """
    Detect simple standalone page-number lines.
    """

    line = normalize_line(line)

    return bool(
        re.match(
            r"^(page\s*)?\d+$",
            line,
            flags=re.IGNORECASE
        )
    )


def remove_repeated_lines(raw_text, repeated_lines):
    """
    Remove repeated lines and standalone page numbers.

    Repeated lines are removed conservatively.
    """

    cleaned_lines = []

    for line in raw_text.splitlines():

        normalized = normalize_line(line)

        if not normalized:
            continue

        comparison_line = normalized.lower()

        if comparison_line in repeated_lines:
            continue

        if is_page_number(normalized):
            continue

        cleaned_lines.append(normalized)

    return "\n".join(cleaned_lines)


# ==================================================
# PAGE CLEANING + SECTION EXTRACTION
# ==================================================

def clean_page_text(
    raw_text,
    repeated_lines,
    current_section
):
    """
    Clean one page and identify section metadata.
    """

    raw_text = fix_hyphenation(raw_text)

    raw_text = remove_repeated_lines(
        raw_text,
        repeated_lines
    )

    raw_text = normalize_whitespace(raw_text)

    if not raw_text:
        return "", current_section

    lines = raw_text.splitlines()

    cleaned_lines = []

    for line in lines:

        line = normalize_line(line)

        if not line:
            continue

        current_section = detect_section(
            line,
            current_section
        )

        cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines)

    cleaned_text = normalize_whitespace(cleaned_text)

    return cleaned_text, current_section


# ==================================================
# CHUNK CREATION
# ==================================================

def create_child_chunks(text):
    """
    Create overlapping child chunks.
    """

    words = text.split()

    chunks = []

    start = 0

    step = CHILD_CHUNK_SIZE - CHILD_OVERLAP

    if step <= 0:
        raise ValueError(
            "CHILD_OVERLAP must be smaller than "
            "CHILD_CHUNK_SIZE."
        )

    while start < len(words):

        end = start + CHILD_CHUNK_SIZE

        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        start += step

    return chunks


def create_hierarchical_chunks(text):
    """
    Create non-overlapping parent chunks.
    """

    words = text.split()

    parent_chunks = []

    start = 0

    while start < len(words):

        end = start + PARENT_CHUNK_SIZE

        parent_text = " ".join(words[start:end]).strip()

        if parent_text:
            parent_chunks.append(parent_text)

        start = end

    return parent_chunks


# ==================================================
# CLEANED TEXT OUTPUT
# ==================================================

def save_cleaned_page(
    pdf_name,
    page_number,
    cleaned_text
):
    """
    Save cleaned page text for manual inspection.
    """

    output_path = (
        CLEANED_TEXT_DIR
        / f"{Path(pdf_name).stem}_cleaned.txt"
    )

    with open(
        output_path,
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            f"\n\n{'=' * 70}\n"
            f"PAGE {page_number}\n"
            f"{'=' * 70}\n\n"
        )

        file.write(cleaned_text)

        file.write("\n")


# ==================================================
# PDF INGESTION
# ==================================================

def load_documents():

    documents = []

    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    print(f"\nFound {len(pdf_files)} PDF files.\n")

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {DOCUMENTS_DIR}"
        )

    ingestion_report = []

    for document_number, pdf_path in enumerate(
        pdf_files,
        start=1
    ):

        document_id = f"DOC{document_number:02d}"

        print(f"Processing: {pdf_path.name}")

        pages = extract_pages(pdf_path)

        repeated_lines = collect_repeated_lines(pages)

        # Start a fresh cleaned-text file for this PDF.
        cleaned_output_path = (
            CLEANED_TEXT_DIR
            / f"{pdf_path.stem}_cleaned.txt"
        )

        if cleaned_output_path.exists():
            cleaned_output_path.unlink()

        current_section = "Unknown"

        document_page_count = 0
        document_cleaned_words = 0
        document_removed_repeated_lines = len(
            repeated_lines
        )

        for page_data in pages:

            page_number = page_data["page"]
            raw_text = page_data["raw_text"]

            cleaned_text, current_section = clean_page_text(
                raw_text,
                repeated_lines,
                current_section
            )

            if not cleaned_text:
                continue

            document_page_count += 1
            document_cleaned_words += len(
                cleaned_text.split()
            )

            save_cleaned_page(
                pdf_path.name,
                page_number,
                cleaned_text
            )

            # ------------------------------------------
            # Create parent chunks
            # ------------------------------------------

            parent_chunks = create_hierarchical_chunks(
                cleaned_text
            )

            for parent_number, parent_text in enumerate(
                parent_chunks,
                start=1
            ):

                parent_id = (
                    f"{document_id}_"
                    f"P{page_number:02d}_"
                    f"PAR{parent_number:02d}"
                )

                # --------------------------------------
                # Create child chunks
                # --------------------------------------

                child_chunks = create_child_chunks(
                    parent_text
                )

                for child_number, child_text in enumerate(
                    child_chunks,
                    start=1
                ):

                    chunk_id = (
                        f"{parent_id}_"
                        f"CH{child_number:02d}"
                    )

                    documents.append({

                        # Text used for embedding
                        "text": child_text,

                        # Source information
                        "source": pdf_path.name,
                        "document_id": document_id,
                        "page": page_number,

                        # Basic structure information
                        "section": current_section,

                        # Hierarchical information
                        "parent_id": parent_id,
                        "chunk_id": chunk_id,

                        # Parent context
                        "parent_text": parent_text

                    })

        ingestion_report.append({

            "document_id": document_id,
            "source": pdf_path.name,
            "total_extracted_pages": len(pages),
            "cleaned_pages": document_page_count,
            "cleaned_word_count": document_cleaned_words,
            "repeated_lines_detected": document_removed_repeated_lines

        })

        print(
            f"  Extracted pages: {len(pages)}"
        )

        print(
            f"  Cleaned pages: {document_page_count}"
        )

        print(
            f"  Repeated lines detected: "
            f"{document_removed_repeated_lines}"
        )

    report_path = REPORTS_DIR / "ingestion_report.json"

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            ingestion_report,
            file,
            ensure_ascii=False,
            indent=2
        )

    return documents


# ==================================================
# CREATE EMBEDDINGS + FAISS
# ==================================================

def create_vectorstore(documents):

    if not documents:
        raise ValueError(
            "No chunks were created. "
            "Check the PDF extraction and cleaning."
        )

    print("\nLoading BGE-M3...")

    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [
        document["text"]
        for document in documents
    ]

    print(
        f"Creating embeddings for "
        f"{len(texts)} child chunks..."
    )

    embeddings = model.encode(
        texts,
        batch_size=8,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    print(
        "\nEmbedding shape:",
        embeddings.shape
    )

    # ----------------------------------------------
    # FAISS INDEX
    # ----------------------------------------------

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    index_path = VECTORSTORE_DIR / "index.faiss"

    faiss.write_index(
        index,
        str(index_path)
    )

    # ----------------------------------------------
    # METADATA
    # ----------------------------------------------

    metadata_path = VECTORSTORE_DIR / "metadata.json"

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            documents,
            file,
            ensure_ascii=False,
            indent=2
        )

    # ----------------------------------------------
    # FINAL INFORMATION
    # ----------------------------------------------

    print("\n" + "=" * 60)
    print("VECTOR DATABASE CREATED SUCCESSFULLY")
    print("=" * 60)

    print(f"FAISS index       : {index_path}")
    print(f"Metadata          : {metadata_path}")
    print(f"Total child chunks: {len(documents)}")
    print(f"Vector dimension  : {dimension}")


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    documents = load_documents()

    print(
        f"\nTotal hierarchical child chunks created: "
        f"{len(documents)}"
    )

    create_vectorstore(documents)

    print("\nIngestion completed successfully.")