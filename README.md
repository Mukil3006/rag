# SLM-RAG Evidence Verification

An existing-document RAG project with local evidence verification **before** final generation.

## Set up from GitHub (Windows)

The repository includes the application, original PDFs, existing FAISS index, tests, and measured reports. Model weights, the Python environment, and the inference runtime are downloaded locally; they are not stored in Git.

```powershell
git clone https://github.com/Mukil3006/rag.git
cd rag
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe setup_models.py
.\.venv\Scripts\python.exe app.py
```

Setup downloads roughly 5 GB of pinned model weights plus the Windows Vulkan runtime. A compatible GPU/driver is needed for the default GPU settings. See `src/config.py` for configuration. Once installed, run only `app.py` using the project environment. The existing FAISS index does not need rebuilding.

Reports contain the actual measurements from the development PC. Absolute trace paths in historical reports refer to that PC; rerun evaluation to generate complete traces under your own checkout.

## Run the application

In VS Code, open this folder and use **Terminal → Run Build Task** (Ctrl+Shift+B), or run:

```powershell
cd C:\SLM\_RAG\_Project
.\.venv\Scripts\python.exe app.py
```

`app.py` starts the configured local model if necessary, starts the web interface, and opens **http://127.0.0.1:8501**. No separate model-start command or API key is required. If launched with another installed Python, it automatically switches to this project's `.venv`. Running it a second time opens the already-running application.

Enter a question and choose **Run verification**. The first request loads BGE-M3 and the reranker; subsequent requests are faster. The page displays the actual final answer, evidence decisions, parent expansion, and complete terminal output. Text and JSON downloads are exact run records. Previous runs are labeled as saved results.

Stop the web app with Ctrl+C if it was started in your terminal. The local model service remains available for CLI experiments. Set `APP_OPEN_BROWSER=0` to suppress automatic browser opening.

## Project structure

```text
app.py                         Single application entry point
src/
  config.py                    Central configuration and environment overrides
  local_runtime.py             Automatic local model startup
  ingestion.py                 Original PDF/chunk/embedding pipeline
  validate_ingestion.py         Original hierarchy validator
  retrieval.py                 BGE-M3 + original FAISS index
  reranking.py                 MiniLM cross-encoder
  verifier.py                  Child/parent and cross-source verification
  evidence_gate.py             Approved evidence boundary
  generation.py                Configurable final generator
  model_client.py              Local/Ollama/API adapters
  pipeline.py                  End-to-end CLI and exact trace formatter
  evaluation.py                Actual A/B/C experiments
  demo.py                      Three live demonstrations
  integration_tests.py         Six real-model test categories
web/                           Browser HTML, CSS, and JavaScript
tests/                         Unit, boundary, and web tests
data/documents/                Original PDFs
data/cleaned_text/              Original extracted text
data/reports/                   Current evaluation and reproducible run records
data/logs/                      Runtime logs and process IDs
vectorstore/                   Preserved index.faiss and metadata.json
models/                        Active BGE-M3, reranker, and Qwen3 weights
runtime/llama/                 Portable local inference runtime
.venv/                         Existing working Python environment
.vscode/                       Interpreter selection and run task
evaluation_questions.json      Original 30 questions, unchanged
evaluation_annotations.json    Separate evaluation-only reference labels
setup_models.py                Model/runtime download utility for setup
requirements*.txt              Dependencies and working-environment lock
```

Obsolete trials, download logs, old backups, and the unused model are preserved outside this project under `C:\SLM\_RAG\_Project_Archive`. See `data/reports/cleanup_manifest.json` for the exact archive location and moved paths. The Git history is preserved.

## Architecture

Query → BGE-M3 → FAISS candidates → cross-encoder reranking → child SLM inspection → conditional parent expansion and re-inspection → cross-source conflict inspection → evidence gate → final generation.

- SUPPORTED: allow grounded source excerpts.
- PARTIAL: allow with an explicit missing-information warning.
- INSUFFICIENT or invalid verification: block.
- CONTRADICTORY: retain both grounded claims and explicitly report disagreement.

The SLM selects exact source sentence IDs; quotations are resolved from the original text. Question-only sentences and isolated numbering are not selectable factual support. Evaluation rubrics and commands addressed to a model are excluded by content before inspection; their original text remains in retrieval logs, with excluded sentence IDs recorded. This does not use document names or expected financial values. Every passage still goes through the real SLM. Invalid selections receive one corrective retry and otherwise fail closed. Child and parent judgments remain separate from the later cross-source verdict.

Generic comparison questions also retrieve their subject to avoid topic dilution. No source names or expected values trigger contradiction labels. Printed test instructions inside the adversarial PDF remain untrusted evidence. Original PDFs, hierarchical metadata, and the 715-vector, 1024-dimensional FAISS index are not regenerated.

## Accuracy and evaluation

An SLM label is not proof that a document is true. Confidence is uncalibrated, and small models can still make semantic mistakes. The project refuses unsupported evidence and exposes conflicts; it does not promise a particular accuracy for arbitrary questions.

Use the actual current results in `data/reports/evaluation_report.json` and `.txt`. Retrieval metrics are expected-source hit rate, reciprocal source rank, and literal keyword coverage. Verification accuracy is scored only on the explicitly annotated subset; per-chunk gold labels and held-out human evaluation are still needed for research conclusions. Do not call keyword overlap semantic accuracy or claim research novelty from implementation success.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe src\integration_tests.py
.\.venv\Scripts\python.exe src\demo.py
.\.venv\Scripts\python.exe src\evaluation.py
.\.venv\Scripts\python.exe src\validate_ingestion.py
```

Modes: **A** basic RAG; **B** reranked RAG; **C** proposed verified RAG. The web selector and CLI `--mode A|B|C` use the same implementation. Full experiments preserve each mode/question trace under `data/reports/experiments/`. Run IDs, exact evidence, scores, raw model outputs, configuration, timestamps, and hashes support reproduction.

## Configuration and hardware

All defaults are in `src/config.py`. Environment overrides include `RETRIEVAL_TOP_K`, `RERANK_TOP_K`, `SLM_MODEL`, `SLM_GGUF_PATH`, `SLM_URL`, `VERIFICATION_RETRIES`, `LLM_PROVIDER`, `LLM_MODEL`, `LLM_URL`, `WEB_UI_PORT`, and `APP_OPEN_BROWSER`.

This PC uses Qwen3 4B Instruct 2507 Q4_K_M through llama.cpp/Vulkan on the RTX 5060 Ti (8 GB VRAM). BGE-M3 and MiniLM run on CPU. Local verification and final generation are separate calls even when sharing model weights. Final generation can alternatively use Ollama, an OpenAI-compatible provider, or explicitly labeled mock mode. External providers are opt-in; keep credentials only in environment variables.

Preserve the exact compatible BGE-M3 embedding model. Model revisions and checksums are in `models/manifest.json`. On another PC, create a Python 3.12 environment and install the dependencies; copied Windows virtual environments are not portable. The active models and runtime are required project dependencies, not disposable files.
