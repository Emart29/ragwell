# Ragwell

A Retrieval-Augmented Generation pipeline with five search strategies, semantic chunking, and **citations that are verified rather than emitted** — every claim carries the chunk it came from and the words that support it, checked by string containment before you see it.

Built with FastAPI, ChromaDB and Jina AI embeddings, and benchmarked against long context on four regulatory annual reports.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

Ragwell ingests documents, embeds them, and searches across them with five strategies — from keyword matching to hypothetical document expansion. On top of that sits an answer layer that does the part most RAG systems skip: it makes every claim quote its source verbatim, then **checks that the quote is really there** before you see it.

That check is a string comparison, so it costs nothing and runs on every answer rather than on a sample. The system can also decline — "the corpus does not support an answer" is a first-class response, because a system with no way to say that will say something else instead.

Both halves are measured. The [benchmark](#what-was-measured) puts retrieval against long context on four regulatory annual reports and reports what each costs and where each gets things wrong.

---

## Citations you can check

Ragwell retrieves. This layer answers, and every claim it makes carries the
chunk it came from and the words that support it — checked, not merely emitted.

![A verified claim with its quote highlighted in the source passage](docs/screenshots/verified-answer.png)

Two things make that answer harder than it looks. Three of the four reports in
the corpus state the **superseded** N200,000 limit, and state it as plainly as
the current report states the new one. And the quote is verified by string
containment against the cited chunk — no second model needed to check it.

The badge is not decoration. It records that those words were found in the
chunk the claim cites, by a comparison you can repeat yourself.

### It can also decline

A system with no way to say "the documents do not support an answer" will say
something else instead, and that something is the failure everyone worries
about. So the decline is a first-class response, not an error path:

![A declined answer, with confidence marked as not applicable](docs/screenshots/declined-answer.png)

![Asking a question, seeing the answer verified, then a question the corpus cannot answer](docs/screenshots/walkthrough.gif)

### What was measured

Four Nigeria Deposit Insurance Corporation annual reports (2020–2024), sixteen
hand-labelled questions, both arms on `gemini-3.5-flash-lite`, 29 August 2026.
Raw results are committed as `benchmark_results.json`, `chunking_results.json`
and `calibration_results.json`.

**Retrieval costs a flat rate; long context scales with the corpus.**

| corpus | retrieval | long context | tokens/question, retrieval | tokens/question, long context |
|---|---|---|---|---|
| 1 report | 100% | 100% | 2,654 | 59,544 |
| 2 reports | 88% | 50% | 2,563 | 115,621 |
| 3 reports | 75% | 86% | 2,557 | 181,981 |
| 4 reports | 88% | 62% | 2,529 | 319,771 |

At four reports long context costs **126×** more per question for the same
question. Retrieval's cost does not move as the corpus grows; that is the whole
point of retrieval and it is the one result here that is arithmetic rather than
sampling.

**Read the accuracy column with its n.** Ten questions per cell means one
question is ten points, so a gap that size is noise. The accuracy comparison is
inconclusive at this sample size and is reported as such.

**Long context does not solve superseded documents.** This is the result that
surprised me. Seeing all four reports at once ought to let a model resolve a
contradiction that retrieval, seeing a slice, cannot. It does not: long context
stated a superseded figure on **67%** of the misleading questions at two of the
four corpus sizes, against retrieval's 33%. An earlier, incomplete run suggested
the opposite and I nearly published it.

**Smaller chunks buy citation precision for free.**

| chunk size | correct | tokens a reader must check |
|---|---|---|
| 128 | 75% | 103 |
| 256 | 75% | 193 |
| 512 | 88% | 401 |
| 1024 | 75% | 819 |

A citation at 1024 tokens asks the reader to check **7.9×** the text of one at
128, and accuracy is flat across the sweep within one question.

**The confidence score is not a probability.** Expected calibration error 15.1%
over 45 outcomes. It is well behaved at the top (+5% in both high bins) and badly
*under*confident at the bottom: answers scored 0.4–0.6 were right every time. No
threshold pays for itself — the best available reaches 93% accuracy by declining
19 of 33 answers to avoid roughly 2 wrong ones. So the fallback threshold is left
unset, on the evidence, and the curve is published instead of a number.

### Using it

```bash
python eval/fetch_corpus.py          # download the corpus (~450MB, checksummed)
python scripts/run_benchmark.py      # the matrix, both arms, four corpus sizes
python scripts/run_chunking_sweep.py # chunk size against citation precision
python scripts/run_calibration.py    # confidence against measured accuracy
python scripts/build_report.py --out docs/report.html   # one self-contained HTML file
```

The three results files are committed, so `build_report.py` rebuilds the report
from the measured data **without an API key and without the corpus**. Every
number quoted above can be checked that way.

Ask a question and see the evidence:

```bash
python scripts/ask.py "What is the maximum deposit insurance coverage?" --judge
```

Or over HTTP:

```bash
curl -X POST localhost:8000/api/answer \
  -H "Content-Type: application/json" \
  -d '{"question": "...", "chunks_shown": 5}'
```

`top_k` is the candidate pool the search returns; **`chunks_shown` is what
reaches the model** after reranking. They are named separately because conflating
them overstates the context by a third, which the earlier metadata in this repo
did.

### How a citation is checked

Three rungs, in increasing cost, each reported separately so the cheap ones can
be counted alone:

1. **Quote containment** — is the quote really in the cited chunk? Free,
   deterministic, and whitespace-normalised because models reflow it when quoting.
2. **Lexical overlap** — do the claim's content words appear in the chunk?
   Catches a genuine quote supporting an overreaching claim.
3. **Entailment** — does the chunk support the claim? One model call, and the
   only rung that costs anything.

The entailment judge is a model judging a model, so its agreement with hand
labels is measured before any number it produces is quoted: **8 of 8** on the
labelled set, zero false supports, stable across three runs. Eight cases is small
and is quoted that way.

### Honest limits

- **Provider-, model- and date-specific.** Measured on 29 August 2026. Two models
  used earlier in this work were withdrawn by their provider mid-benchmark.
- **Ten questions per cell.** Strong enough for the cost and trap findings, not
  for accuracy differences.
- **The corpus is one regulator's annual reports.** Dense, numeric, and English.
  A corpus of prose would behave differently.
- **The scripts reset the index.** Each measurement script clears the store and
  re-ingests at the size it needs, so run one and the demo corpus is gone until
  you re-ingest.
- **ChromaDB does not see another process's writes.** Ingest, then restart the
  server, or it will answer from a stale view.

---

## Architecture

**The request path.** A question becomes claims, and every claim is checked
against the chunk it cites before it reaches the reader.

```mermaid
graph TB
    CLIENT["Answer UI · ask.py · benchmark harness"] --> ANSWER["POST /api/answer"]
    ANSWER --> RETRIEVE["Retrieval — vector, keyword FTS5,<br/>hybrid, HyDE, query expansion"]
    RETRIEVE --> RERANK["Jina reranker<br/>top_k candidates → chunks_shown"]
    RERANK --> GEN[Generator]

    CONTRACT["Contract: a claim must cite a chunk<br/>and quote it verbatim.<br/>An uncited claim cannot be constructed."] --> GEN

    GEN --> LADDER["Verification ladder<br/>1 · quote containment — free<br/>2 · lexical overlap — free<br/>3 · entailment — one model call"]
    LADDER --> CONF["Confidence from retrieval strength,<br/>verification, coverage, agreement"]
    CONF --> DECIDE{"Evidence<br/>sufficient?"}
    DECIDE -->|yes| OUT["Answer with checked citations"]
    DECIDE -->|no| DECLINE["InsufficientEvidence<br/>a first-class answer"]

    VERIFY["POST /api/answer/{id}/verify"] --> LADDER

    style CONTRACT fill:#fff4e5,stroke:#d68910
    style LADDER fill:#e8f6ef,stroke:#1e8449
    style DECLINE fill:#fdedec,stroke:#c0392b
```

**Ingestion, storage and measurement.** Chunk ids are derived from content, so a
citation survives re-ingestion.

```mermaid
graph LR
    INGEST["POST /api/ingest"] --> PIPE["Parse → chunk → validate<br/>content-derived chunk ids"]
    PIPE --> EMBED["Jina embeddings<br/>+ cache"]
    EMBED --> SQLITE[(SQLite + FTS5)]
    EMBED --> CHROMA[(ChromaDB)]

    BENCH[Benchmark harness] --> RAG["RAG arm<br/>retrieve, then answer"]
    BENCH --> LONG["Long-context arm<br/>same contract, no retrieval"]
    RAG --> SCORE["Scorer — correctness, overreach,<br/>superseded-figure rate, cost"]
    LONG --> SCORE
    SCORE --> REPORT["Self-contained HTML report"]

    SQLITE -.-> RAG
    CHROMA -.-> RAG
```

The retrieval half is ragwell as it was. The contract, the ladder, the
long-context arm and the scorer are what the citation work added.

---

## Features

**The answer layer**

- **Verified citations** — every claim cites a chunk and quotes it verbatim, checked three ways
- **Declines when it should** — "the corpus does not support an answer" is a first-class response, not an error
- **Calibrated confidence** — scored from retrieval strength and citation checks, then plotted against measured accuracy rather than asserted
- **Benchmarked against long context** — retrieval versus sending the whole corpus, measured on cost and on correctness

**The retrieval layer**

- **Multi-format ingestion** — PDF, DOCX, TXT, CSV, HTML, Markdown
- **5 search strategies** — Vector, Keyword (FTS5), Hybrid, HyDE, Query Expansion
- **Jina AI integration** — asymmetric embeddings (query vs. passage) and reranking
- **Semantic chunking** — recursive and semantic, with quality validation
- **Background processing** — async ingestion with status tracking

---

## Tech Stack

| Category | Technology |
|---|---|
| Backend | FastAPI, Uvicorn, Python 3.11+ |
| Vector DB | ChromaDB |
| Relational DB | SQLite with FTS5 |
| Embeddings | Jina AI Embeddings v3 |
| Reranking | Jina Reranker v2 |
| LLM | Groq and Google Gemini, named explicitly and never substituted |
| Chunking | LangChain Text Splitters |

---

## Performance

| Operation | Latency |
|---|---|
| Keyword search (FTS5) | < 15ms |
| Vector search | ~650ms |
| Hybrid search | ~680ms |
| HyDE | ~2.4s |
| Query expansion | ~3.4s |
| **Answer with citation checks** | **~1.7–2.6s** |
| Answer with the entailment judge | ~4s |

Ingestion runs about 50s for a 262-page PDF. Table extraction is off by default:
`pdfplumber` walks every page a second time and dominates parse time — roughly
six minutes against 29 seconds for the text — while nothing downstream reads the
result. Set `EXTRACT_PDF_TABLES=true` if you need it.

---

## Quick Start

### Prerequisites

- Python 3.11+
- [Jina AI API key](https://jina.ai/embeddings/) — required for embeddings and reranking. Free tier: 10M tokens/month
- [Google AI Studio key](https://aistudio.google.com/apikey) — required for answering. Free tier, no card
- [Groq API key](https://console.groq.com/) — optional second provider

Reproducing the benchmark also needs the corpus: four NDIC annual reports,
about 450MB, downloaded by `python eval/fetch_corpus.py`. The committed results
can be read and the report rebuilt without it.

### Installation

```bash
git clone https://github.com/Emart29/ragwell.git
cd ragwell

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

mkdir -p data/uploads data/chroma

cp .env.example .env
```

### Configuration

Edit `.env` with your API keys:

```env
JINA_API_KEY=jina_xxxxxxxxxxxx   # Required — embeddings and reranking
GEMINI_API_KEY=xxxxxxxxxxxx      # Required — answering. Every published number was measured on it
GROQ_API_KEY=gsk_xxxxxxxxxxxx    # Optional — second provider, and HyDE/query expansion
```

Providers are named and never substituted for one another. An answer produced by
one model is never attributed to another, because the benchmark compares them and
a silent fallback would make the comparison meaningless.

### Start the server

```bash
python -m uvicorn app.main:app --port 8000
```

The API will be running at `http://localhost:8000`. Interactive docs are available at `http://localhost:8000/docs`.

### Ask a question and see the evidence

`ask.py` answers from whatever is in the index, so put something there first.
The quickest way needs no download:

```bash
python scripts/demo.py
```

This generates three sample documents, ingests them, and runs all five search
strategies with retrieval metrics. It starts its own server on port 8000 — so
stop the one above first — and it **clears the index** before it begins.

Then, in a second terminal:

```bash
python scripts/ask.py "What do hybrid search approaches combine?"
```

```
  1. [verified] Hybrid search approaches merge vector and keyword results.
       source: company_report.pdf, p.1  [06482654e493]
       quote: "Hybrid approaches merge vector and keyword results to leverage
              the strengths of both methods."

  confidence 0.95
  citations: 1 claims, quote found 100%, decorative 0%
```

`[verified]` means the quote was found in the cited chunk by string comparison.
Add `--judge` to run the entailment rung, `--show-chunks` to print the cited
text. Open `http://localhost:8000` for the same thing in a browser, where
clicking a claim highlights its quote in the passage.

To ask against the regulatory corpus the benchmark uses instead:

```bash
python eval/fetch_corpus.py                    # four NDIC annual reports, ~450MB
python scripts/run_benchmark.py --sizes 1      # ingests, then measures
python scripts/ask.py "What is the maximum deposit insurance coverage?" --judge
```

**Two things to know.** Every measurement script resets the index and re-ingests
at the size it needs, so running one discards whatever was there. And ChromaDB
does not see another process's writes — ingest, then restart the server, or it
will answer from a stale view.

---

## API Reference

Base URL: `http://localhost:8000/api`

### Health check

```http
GET /api/health
```

```json
{
  "status": "healthy",
  "total_documents": 10,
  "total_chunks": 150,
  "chromadb_collection_count": 150
}
```

### Upload a document

```http
POST /api/ingest
Content-Type: multipart/form-data
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | file | — | Document to upload (PDF, DOCX, TXT, CSV, HTML) |
| `chunk_strategy` | string | `recursive` | `recursive` or `semantic` |
| `chunk_size` | int | `512` | Characters per chunk (128–1024) |
| `chunk_overlap` | int | `50` | Overlap between chunks (0–200) |

```json
{
  "document_id": "uuid",
  "status": "processing",
  "filename": "document.pdf"
}
```

### Query documents

```http
POST /api/query
Content-Type: application/json
```

```json
{
  "query": "What is machine learning?",
  "strategy": "hybrid",
  "top_k": 10,
  "rerank": true,
  "alpha": 0.7
}
```

`alpha` controls the vector/keyword balance in hybrid search (0 = keyword only, 1 = vector only).

```json
{
  "results": [
    {
      "chunk_id": "uuid",
      "text": "Machine learning is...",
      "score": 0.85,
      "document": {
        "id": "uuid",
        "filename": "ml_guide.pdf"
      }
    }
  ],
  "strategy": "hybrid",
  "search_time_ms": 650.5,
  "total_candidates": 20
}
```

### List documents

```http
GET /api/documents?page=1&per_page=20
```

### Get document chunks

```http
GET /api/documents/{document_id}/chunks
```

### Delete a document

```http
DELETE /api/documents/{document_id}
```

---

## Search Strategies

| Strategy | How it works | Best for |
|---|---|---|
| `keyword` | SQLite FTS5 full-text search | Exact term matching, fast lookups |
| `vector` | Semantic similarity via ChromaDB | Conceptual or paraphrased queries |
| `hybrid` | Weighted fusion of vector + keyword | General-purpose (recommended default) |
| `hyde` | Generates a hypothetical answer, embeds it, searches | Complex or abstract questions |
| `expanded` | LLM rewrites the query into multiple variants, searches all | Broad topic exploration |

---

## Project Structure

```
ragwell/
├── app/
│   ├── api/
│   │   ├── ingest.py          # Document upload endpoints
│   │   ├── query.py           # Search endpoints
│   │   └── documents.py       # Document management endpoints
│   ├── embeddings/
│   │   └── embedder.py        # Jina AI embedding integration
│   ├── retrieval/
│   │   ├── vector_search.py   # ChromaDB vector search
│   │   ├── keyword_search.py  # FTS5 keyword search
│   │   ├── hybrid_search.py   # Combined search with score fusion
│   │   ├── reranker.py        # Jina Reranker integration
│   │   └── query_expand.py    # HyDE and query expansion
│   ├── storage/
│   │   ├── database.py        # SQLite connection management
│   │   ├── models.py          # ORM models
│   │   └── vector_store.py    # ChromaDB wrapper
│   ├── parsing/               # Document parsers (PDF, DOCX, etc.)
│   ├── chunking/              # Text chunking strategies
│   ├── validation/            # Chunk quality validation
│   ├── llm/                   # LLM provider abstractions
│   └── main.py                # FastAPI app entry point
├── scripts/
│   └── demo.py                # End-to-end demo script
├── evaluation/                # Evaluation and benchmarking tools
├── tests/
├── data/                      # Runtime data (uploads, ChromaDB)
├── requirements.txt
└── .env.example
```

---

## Testing

```bash
# Fast unit tests — no API keys required (~25s)
python -m pytest tests/test_parsers.py tests/test_cache.py tests/test_quality.py -v

# With coverage report
python -m pytest tests/test_parsers.py tests/test_cache.py tests/test_quality.py --cov=app --cov-report=html

# Full test suite including API integration tests (~5 min, requires API keys)
python -m pytest tests/ -v --timeout=60
```

| Suite | Tests | Time | Requires API keys |
|---|---|---|---|
| Parsers | 30 | ~6s | No |
| Cache | 10 | ~1s | No |
| Quality | 7 | ~1s | No |
| API Integration | 17 | ~5min | Yes (Jina + Groq) |

The fast unit tests are recommended for CI/CD pipelines.

---

## Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t ragwell .
docker run -p 8000:8000 --env-file .env ragwell
```

### Docker Compose

```yaml
version: '3.8'
services:
  ragwell:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - ./data:/app/data
```

### Scaling notes

When moving beyond a single instance, a few things to consider:

- **Database**: Replace SQLite with PostgreSQL (supports concurrent writes and multiple instances).
- **Vector store**: For large datasets (millions of chunks), consider a dedicated vector DB like Pinecone or Weaviate instead of ChromaDB.
- **Caching**: Add Redis to cache frequent query results and reduce embedding API calls.
- **Job queue**: Use Celery + Redis to manage document processing jobs across workers.

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [Jina AI](https://jina.ai) for embeddings and reranking APIs
- [Groq](https://groq.com) for fast LLM inference
- [ChromaDB](https://www.trychroma.com) for vector storage
- [FastAPI](https://fastapi.tiangolo.com) for the API framework