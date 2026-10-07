# RAG Basics & Pipeline Architecture

## Core Concept
**RAG (Retrieval-Augmented Generation)** enhances LLM responses by retrieving relevant information 
from a private/external knowledge base and injecting it into the prompt context.

---

## Vector Database Record Schema
```json
{
  "id": "chunk_01",
  "text": "Exact text snippet content...",
  "vector": [0.012, -0.045, 0.891 ],
  "metadata": {"source": "manual.pdf", "page": 12}
}
```

## RAG Pipeline Overview

Documents ──> Chunking (with overlap) ──> Embedding Model ──> Vector DB
                                                                  │
User Query ──> Embedding Model ──> Similarity Search (Cosine) ────┘
                                         │
                                   Top-K Chunks
                                         │
                                  Prompt Injection ──> LLM ──> Answer

## Critical Engineering Takeaways

- **Chunking Quality > Model Size**: Naive character-based splitting breaks semantics. Splitting by logical structures (paragraphs, sentences) is essential for retrieval precision.

- **Semantic vs. Keyword Search**:

    - **Vector Search**: Excellent for conceptual similarity, synonyms, and natural language query mapping.

    - **Keyword Search (BM25)**: Critical for exact matches (SKUs, IDs, part numbers like F12).

- **Hybrid Search**: Combines Vector Search and BM25 to maximize coverage and accuracy in production systems.


---------

# Hybrid Search & Reranking Architecture

## Overview
Standard Vector Search can miss exact matches (e.g., product IDs, acronyms, specific terms), 
while Keyword Search fails to capture semantic meaning. Combining both techniques via **Hybrid Search** 
and applying a **Reranker** provides optimal retrieval performance in production RAG pipelines.

---

## Pipeline Architecture

```text
User Query
   │
   ├───> 1. Vector Search (Semantic)   ──> Top-K Candidates (by Cosine Similarity)
   │                                           │
   └───> 2. BM25 Search (Keyword)     ──> Top-K Candidates (by Exact Matches)
                                               │
                                               ▼
                               3. Reciprocal Rank Fusion (RRF)
                                               │
                                               ▼
                                   Combined Top-N Candidates
                                               │
                                               ▼
                                    4. Cross-Encoder / Reranker
                                               │
                                               ▼
                                      Final Top-3 Context
                                               │
                                               ▼
                                        LLM Generation

```

## Core Components
1. **BM25 (Keyword Search)**
- Purpose: Matches exact terms, product codes, abbreviations, and specific identifiers.

- Mechanism: Statistical algorithm based on Term Frequency (TF) and Inverse Document Frequency (IDF).

- Tokenization: Requires explicit tokenization/stemming suited to the target language.

2. **Reciprocal Rank Fusion (RRF)**
- Purpose: Merges rank lists from different retrieval methods without normalizing raw scores
(which operate on incompatible mathematical scales).

- Formula:  $RRF\_Score(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$
    - $r_m(d)$: Rank of document $d$ in retrieval method $m$.
    - $k$: Smoothing constant (typically $k = 60$).

3. Reranking (Cross-Encoder / LLM Reranker)Purpose: Deeply evaluates query-document pairs 
to filter out false positives and select the absolute best candidates for prompt injection.
Mechanism:Bi-Encoder (Vector DB): Fast retrieval ($O(1)$ search over pre-computed vectors).
Cross-Encoder / LLM Reranker: Slow but highly accurate evaluation of $N$ candidates jointly with the query.

| Stage | Speed | Accuracy | Primary Use Case |
| --- | --- | --- | --- |
| **Vector Search** | Fast | High (Semantic) | Concepts, synonyms, intent matching |
| **BM25 Search** | Very Fast | High (Exact) | Part numbers, names, rare terms |
| **RRF Fusion** | Instant | N/A | Score-agnostic rank unification |
| **Reranker** | Slower | Max Precision | Final Top-K filtering for LLM context |
