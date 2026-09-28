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
  "vector": [0.012, -0.045, 0.891, ...],
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