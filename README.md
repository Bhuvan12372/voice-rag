
## Scaling to a 500-page PDF

The larger PDF increases one-time indexing work and local retrieval work, but
the llm model never receives the whole document:

| Work | How it scales |
|---|---|
| Ingest | Parsing, chunking, and embedding grow with pages/chunks. The index is cached by PDF hash, so this is not repeated for every question. |

| LLM input and API cost | The final context is capped at 7 chunks and conversation history is bounded. The pipeline normally makes 2 LLM calls and has a fixed maximum of 4 on the not-found retry path. Thus prompt size and model-call count do not grow with PDF length. |

| Dense retrieval | Chroma returns at most 20 nearest chunks using an approximate nearest-neighbor index; it does not send every chunk to the reranker. |

| BM25 retrieval | The in-memory implementation scores and sorts the indexed chunks, so its work grows with corpus size. For substantially larger corpora, replace it with an indexed sparse-search service rather than claiming constant retrieval time. |

| Main reranking | RRF passes at most 12 candidates per query. The standalone question plus at most 3 sub-questions bounds the normal workload to 4 query-specific batches (at most 48 candidate/query pairs), independent of the number of PDF pages. |

The update-override pass also reranks update candidates. It normally uses up
to 10 nearest update chunks per query, but explicit references to an update
chapter can currently add every update chunk from that chapter. If a 500-page
document has a very large update chapter, that path can grow; cap or index
those candidates before making a strict latency guarantee.

So the design keeps **LLM token cost and normal reranker work roughly constant**,
while local retrieval and first-time indexing still grow with corpus size.
The measured evaluation uses the 11-page sample PDF; a 500-page latency claim has not been benchmarked. Larger documents also increase the chance of near-duplicate or conflicting chunks, so retrieval quality should be rechecked on representative documents.


# Voice Study Assistant

Ask a question out loud about a PDF and get a short, cited answer, with the cost and time of every question measured and shown.

* **Ingest once, ask unlimited questions.** The PDF is parsed, chunked, embedded and cached on disk, keyed by its hash.
* **Grounded answers only.** The model sees just the retrieved chunks. If the document does not contain the answer, the app says so.
* **Every answer cites pages**, and the citations are verified in code.

> The brief and the sample PDF are confidential, so `data/*.pdf` is git-ignored. Put the PDF in `data/` yourself.

## Quick start

```bash
pip install -r requirements.txt

cp /path/to/Kestrel_Islands_Field_Guide.pdf data/

streamlit run app.py            # allow microphone access
```

Create a project-root `.env` file with your provider keys:

```dotenv
LLM_API_KEY=your-answer-model-key
GROQ_API_KEY=your-groq-api-key
```

Speech is transcribed with Groq's `whisper-large-v3`; this is separate from the
`LLM_API_KEY` used for question answering. You can still type questions without
a Groq key. If Groq is not configured or its transcription request fails, a
browser speech-recognition option appears (Chrome recommended). Browser
recognition is not billed through your Groq key, but it is a browser-provided
service and its availability or limits are not guaranteed. Never commit `.env`
or share either key.

The first start downloads the two local models (~500 MB) and builds the index (seconds). Later starts load the cached index.

```bash
python -m eval.run_eval         # real run of the question set -> eval/results.md
```

## How it works

```
INGEST (once; cached in cache/<hash>/)
 PDF -> pdfplumber: text lines with font info + tables
     -> drop the repeated header/footer (found automatically, not hard-coded)
     -> headings by font size -> chapter / section
     -> chunks: prose per section; each table as markdown WITH its header row
     -> metadata: page, chapter, kind, is_update, see_also
     -> search text = chapter/section labels + chunk text; table rows become plain sentences
     -> BGE embeddings -> Chroma;  chunks.json -> in-memory BM25

QUESTION
 speech (recorded in browser, transcribed by Groq Whisper, shown + editable)
  -> 1: understand   LLM: standalone question, sub-questions, intent      [~1 short call]
  -> 2: retrieve     dense top-20 + BM25 top-20 -> Reciprocal Rank Fusion -> 12 candidates per query
  -> 3: rerank       cross-encoder -> top 5 (top 3 per sub-question), dropping scores below 0.001
  -> 4: override     force relevant "later update" chunks into the context
  -> 5: guard        nothing relevant at all? answer "not in the document", skip the LLM
  -> 6: answer       LLM sees only the labelled chunks, returns JSON {found, answer, pages}
  -> 7: verify       cited pages must be among the pages shown; otherwise rejected
  -> 8: retry        only if "not found": decompose once, retrieve again, answer again
```

### Decisions and why

| Decision | Reason |
|---|---|
| Hybrid search with RRF | Dense search handles paraphrase; BM25 handles exact names and times (`09:15`). Raw scores are on different scales, so fusion is by rank. |
| Cross-encoder reranker (Internally uses transformer architecture)| Reads query and chunk together, so it picks the chunk that *answers* the question, not just one on the same topic. |
| No MMR | This document needs near-duplicate chunks together: the old ferry table and the later update are the same topic, and the update must not be pruned. |

| Tables in two formats | The answer model sees a readable table. For search, each row is written as a sentence, which is easier for the search models to match. |

| Breadcrumb before every chunk (`Chapter 3: … \| The mainland ferry`) | Tables have no title of their own. |
| Metadata written into the prompt (`[Page 5 \| Ch.3 … \| table]`) | 

| Header/footer removed | Otherwise "Kestrel" appears in every chunk and pollutes BM25 and embeddings. |

| One call for rewrite + decomposition | Follow-ups ("Why is that?") and multi-fact questions are both handled in one cheap call. History keeps the last 3 turns **including the previous answer**, because "that" refers to the answer. |

| Retrieval score never makes an answer "found" | A related chunk is not an answer ("hotel prices" can retrieve the fares table). Only the LLM's `found` plus the citation check can do that. The score is used only to *skip* work when it is very low. |

| Bounded steps instead of free-form ReAct | At most 2 LLM calls normally and 4 on the retry path, so cost and latency are predictable and measurable. |

| Hash-keyed cache | The key includes the PDF hash, embedding model and chunking settings/version, so a stale index is never reused silently. |


## Cost and time

LLM calls and Groq Whisper transcription may incur provider charges. Whisper transcription is not included in the per-question LLM cost metrics. Browser speech fallback is not billed through Groq, but browser-service availability and limits may vary. The embedding and reranker models run locally (₹0 API cost, but they take CPU time, which is measured and reported separately as "local compute").

Token counts come from the provider's `usage` field. Prices are in `.env` (`PRICE_*`, `USD_TO_INR`, `PRICING_NOTE`); record the date you copied them.

The selected evaluation run recorded the following for `T1` and `T2` (the
two test cases run) and `D8` (one dev case). These are measured for that run,
not estimates or results for the complete six-question test set.

| Scope | Passed | Mean cost / max | Median / p90 latency |
|---|---:|---:|---:|
| Test subset (`T1`, `T2`) | 2/2 | ₹0.0152 / ₹0.0162 | 2.44 s / 2.84 s |
| Dev subset (`D8`) | 1/1 | ₹0.0147 / ₹0.0147 | 1.89 s / 1.89 s |

The saved question-level report does not include model pricing details,
per-stage timings, not-found retry timing, or one-time ingest time.

## Test results

* **Question-set evaluation (`python -m eval.run_eval`)** runs the real app models. `eval/questions.json` has a 19-question **dev** set (used to choose `RERANK_MIN_SCORE`, `UPDATE_MIN_SCORE` and prompt wording) and a **6-question test** set, which is not used for tuning. The saved run summarized below selected only `T1`, `T2`, and `D8`; it does not report results for all six test questions.

The evaluation runner writes `eval/results.json` with the question, answer,
retrieved/cited pages, latency, cost, and pass/fail checks. It also writes
`eval/results.md` as a readable summary. Running it uses the configured answer
model and may consume its quota. The expected answers and citation checks are
defined in `eval/questions.json`; unit tests separately cover parser,
guardrail, and pipeline behavior.

### Results from the saved selected-case run

| ID | Split | Outcome | Cited pages | Latency | Cost |
|---|---|---|---|---:|---:|
| T1 | Test | PASS: gave the updated 09:15 ferry time | 5, 11 | 2.84 s | ₹0.0162 |
| T2 | Test | PASS: named Agathe Rindel | 4 | 2.05 s | ₹0.0141 |
| D8 | Dev | PASS: gave the €7 levy from April 2026 | 2, 5, 11 | 1.89 s | ₹0.0147 |

`T3`–`T6` were not part of this selected run. Run `python -m eval.run_eval
--set test` to evaluate the full test set; this uses the configured answer
model and may consume quota.

## Limitations (honest list)

* **Speech recognition** records audio in the browser and sends it to Groq's Whisper API; it needs microphone permission, internet access, and a Groq API key. If Groq is unavailable or the request fails, browser speech recognition can be used as a fallback (Chrome recommended). Both can mishear names, so the transcript is shown and editable. Set `WHISPER_MODEL` to change the default `whisper-large-v3`.
* **Update handling is document-specific.** It relies on an updates/errata chapter title. Inline corrections, dated notices, or differently named chapters are not detected. A general solution would need date and supersession parsing, which is unreliable.
* **"Now" is defined by the document, not the calendar.** The guide is the 2025 edition and says the levy rises to €7 from 1 April 2026. The model is told not to assume today's date, so it states both the value and the dated change.
* **Scores and thresholds** were tuned on a small dev set of a single document; treat them as starting points for another document. `RERANK_MIN_SCORE` checks the best result to decide whether to answer, while `CONTEXT_MIN_RERANK_SCORE` (default `0.001`) filters each selected chunk before it reaches the answer model. `UPDATE_MIN_SCORE` gates update-override chunks. `RERANK_MIN_SCORE` checks the best result to decide whether to answer; `CONTEXT_MIN_RERANK_SCORE` filters individual chunks before they reach the answer model.
* **Chunks never cross pages**, so a fact that straddles a page break is split. There is no chunk overlap because sections are self-contained here.
* **PDF layout assumptions:** text-based PDF with ruled tables and consistent heading fonts. Scanned PDFs (no OCR) or unruled tables would need another parser.
* **Not-found questions cost more**: the retry adds a decomposition call and a second answer call (see the table above).
* **Single user, single process**: sessions live in Streamlit session state; no authentication or persistence of history.

## Repository layout

```
app.py                      Streamlit UI (voice in, answer + metrics out)
voicerag/
  config.py                 all settings (env-overridable), CHUNKING_VERSION
  models.py                 dataclasses: Chunk, ScoredChunk, FinalAnswer, ...
  pdf_parser.py             PDF -> headings / paragraphs / tables, header stripping
  chunker.py                blocks -> chunks with metadata
  embedder.py               BGE embeddings (local)
  indexer.py                hash-keyed on-disk cache + Chroma
  retriever.py              BM25, RRF, hybrid retriever
  reranker.py               cross-encoder reranker
  prompts.py                all prompt text
  llm_client.py             OpenAI-compatible client, usage -> cost
  query_understanding.py    follow-up rewrite, decomposition
  generator.py              grounded answer call
  guardrails.py             citation check, relevance floor
  pipeline.py               orchestration (override pass, retry)
  cost_tracker.py           per-call / per-question / per-session accounting
  bootstrap.py              wires real components together
eval/                       questions.json, run_eval.py
tests/                      unit tests with fakes
```
