# RAG Retrieval Benchmark

## 1. Goal

This benchmark evaluates the retrieval layer itself, independently from LLM answer
quality. It is designed to answer concrete engineering questions:

- Does Dense Retrieval retrieve the right evidence?
- Does BM25 recover exact financial/disclosure terminology missed by Dense Retrieval?
- Does RRF improve recall over either single retriever?
- Does the optional Cross-Encoder Reranker improve early-rank quality?
- Does the PIT filter ever return evidence published after the research cutoff?

The benchmark deliberately does **not** claim that higher retrieval metrics imply
better stock returns or better investment decisions.

## 2. Current production retrieval chain

The production retriever is:

```text
Query
  |
Scope resolution
company + industry + market + macro + regulation
  |
PIT filter
publish_date <= as_of_date
  |
+----------------------+----------------------+
|                                             |
Qdrant Dense Retrieval                 Local BM25
FastEmbed BGE embedding                PIT/scope-filtered corpus
|                                             |
+----------------------+----------------------+
                       |
                      RRF
                       |
             Optional Cross-Encoder
                    Reranker
                       |
            Parent-document diversity
                       |
                  Evidence Pack
```

The benchmark uses the same `HybridKnowledgeRetriever` implementation and exposes
four ablation strategies without changing the production default:

- `dense`
- `bm25`
- `hybrid` = Dense + BM25 + RRF
- `hybrid_rerank` = Dense + BM25 + RRF + configured Cross-Encoder

## 3. Why manual ground truth is required

A retrieval metric is only meaningful if there is a human-reviewed
Query -> Relevant Evidence mapping.

The repository therefore does not publish a fabricated Recall@K number. The
starter dataset contains retrieval questions but intentionally contains no
pre-filled relevance labels.

The annotation workflow is pooled across several retrieval strategies to reduce
single-system annotation bias.

## 4. Starter query set

`evaluation/datasets/rag_retrieval_queries_v1.jsonl` contains 45 starter cases:

- 9 actually ingested autumn-recruitment demo companies;
- 5 generic equity-research dimensions per company:
  - business model / core business;
  - capex / construction / capacity plan;
  - profitability / cash flow / balance sheet;
  - operating / industry / policy / financial risks;
  - competitive position / core competitiveness.

The default cutoff is aligned with the initial autumn corpus snapshot. If the
corpus changes, create a versioned query/annotation dataset rather than silently
reusing old labels.

## 5. Prepare the annotation worksheet

Make sure Qdrant is running and the corpus has already been ingested. Then:

```bash
python scripts/run_rag_retrieval_benchmark.py prepare \
  --dataset evaluation/datasets/rag_retrieval_queries_v1.jsonl \
  --output evaluation/data/rag_retrieval_annotations_v1.csv
```

The script pools candidates from Dense, BM25, Hybrid RRF and, when available,
Hybrid + Reranker.

The CSV contains:

```text
case_id
ticker
company_name
dimension
query
as_of_date
rank_sources
chunk_id
doc_id
document_key
file_name
page
publish_date
source_url
excerpt
relevance
notes
```

### Relevance labels

Fill `relevance` manually:

- `0`: not relevant;
- `1`: relevant context;
- `2`: highly relevant evidence;
- `3`: direct evidence that can answer the query;
- blank: not yet judged.

For Recall@K, any label > 0 is considered relevant. nDCG uses the 1/2/3 grades.

Do not mark a row relevant merely because it mentions the company or the same
keyword. It should contain evidence that materially contributes to the answer.

## 6. Run the benchmark

After annotation:

```bash
python scripts/run_rag_retrieval_benchmark.py run \
  --dataset evaluation/datasets/rag_retrieval_queries_v1.jsonl \
  --annotations evaluation/data/rag_retrieval_annotations_v1.csv \
  --output results/rag_retrieval_benchmark_v1 \
  --ks 5,10
```

Outputs:

```text
results/rag_retrieval_benchmark_v1/
├── raw_runs.jsonl
├── per_case.csv
├── summary.csv
└── RAG_RETRIEVAL_REPORT.md
```

Cases with no positive human judgment are skipped instead of being treated as
zero-recall cases.

## 7. Metrics

### Evidence Recall@K

For one query:

```text
Recall@K =
number of annotated relevant evidence units retrieved in top K
/
total annotated relevant evidence units
```

Example:

- 4 relevant evidence units are annotated;
- Top-5 retrieves 3 of them;
- Recall@5 = 3 / 4 = 0.75.

The evaluator performs one-to-one matching so overlapping chunks cannot
double-count the same annotated evidence item.

### Document Recall@K

Chunk-level metrics are sensitive to chunk boundaries. Therefore the benchmark
also reports parent-document recall:

```text
DocumentRecall@K =
relevant parent documents represented in top K
/
all relevant parent documents
```

This is useful when adjacent chunks from the same annual report contain the same
fact.

### Precision@K

```text
Precision@K =
relevant evidence units retrieved in top K / K
```

### HitRate@K

1 if at least one relevant evidence unit appears in Top-K, otherwise 0.

### MRR@K

Reciprocal rank of the first relevant evidence item. It measures how quickly a
useful result appears.

### nDCG@K

Uses the human 1/2/3 relevance grades to evaluate ranking quality, not just binary
retrieval.

### PIT violation rate

```text
future-dated returned chunks / all returned chunks
```

The expected value is 0. A non-zero value indicates a retrieval safety regression.

## 8. Fair ablation rules

When comparing Dense / BM25 / RRF / Reranker, keep the following fixed:

- exact same Qdrant corpus;
- exact same `as_of_date`;
- exact same scope rules;
- same `candidate_k`;
- same BM25 `corpus_limit`;
- same final Top-K;
- same parent-document diversity cap;
- same annotation set.

Only the retrieval/ranking strategy should change.

The runner defaults to the same project settings:

```text
candidate_k = 30
bm25 corpus_limit = 1000
max_chunks_per_doc = 2
RRF k = 60
```

## 9. Annotation-pool bias

Pooled annotation is practical but not perfect. Evidence that none of the tested
retrievers retrieves may never appear in the worksheet.

For a stronger benchmark:

1. sample several cases;
2. inspect the original annual/semiannual/IR documents manually;
3. add missed relevant evidence to the annotation CSV using document_key + page;
4. rerun the benchmark.

This turns the benchmark from "pooled relevance" toward a more complete
evidence-level ground truth.

## 10. Chunking experiments

Current production chunking is:

```text
PDF page parsing
    ->
paragraph-aware splitting
    ->
target ~900 characters
    ->
120-character trailing overlap
```

The retrieval benchmark can later be reused to compare:

- 600 / 900 / 1200-character targets;
- 0 / 120 / 200-character overlap;
- current paragraph-aware chunking;
- heading/section-aware or semantic chunking.

Do not change chunking and retrieval strategy in the same experiment if the goal is
to isolate which component caused the metric change.

## 11. Resume/interview wording

Before manual labels exist:

> Implemented a reproducible RAG retrieval benchmark framework covering Recall@K,
> Document Recall, MRR, nDCG, PIT violations and Dense/BM25/RRF/Reranker ablations.

After a real annotation run, you may report measured results, but only with the
dataset size and settings, for example:

> On N manually annotated A-share disclosure queries, Hybrid RRF achieved X Recall@5
> versus Y for Dense-only under the same corpus/cutoff settings.

Do not publish placeholder values as measured results.
