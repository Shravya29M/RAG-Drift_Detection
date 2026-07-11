# Retrieval-Augmented Generation

Retrieval-augmented generation (RAG) grounds a large language model's answers in an
external document corpus instead of relying only on what the model memorised during
training. At query time the system embeds the user's question, retrieves the most
similar text chunks from a vector index, and injects those chunks into the prompt as
context. The model is instructed to answer strictly from that context, which sharply
reduces hallucination and lets the knowledge base be updated without retraining.

## Ingestion and chunking

Documents arrive as PDFs, Markdown files, plain text, or web pages. Each source is
parsed to raw text and split into overlapping windows of tokens. The overlap between
consecutive chunks preserves sentences that would otherwise be cut at a boundary, so
a fact spanning two chunks remains retrievable from at least one of them. Every chunk
carries provenance metadata: the source document, its position, the page number for
PDFs, and the nearest section heading for Markdown.

## Embedding and indexing

Each chunk is encoded into a dense vector with a sentence-transformer model and
L2-normalised, so the inner product of two vectors equals their cosine similarity.
The vectors are stored in a FAISS index. Queries are encoded with the same model,
and the top-k nearest chunks are returned, optionally filtered by metadata such as
source type or section.

## Generation

Retrieved chunks are formatted into a numbered context block and combined with the
question in a prompt template. The language model produces an answer along with the
supporting sources, so every claim can be traced back to a specific chunk in a
specific document. When retrieval returns nothing relevant, the system says so
explicitly rather than inventing an answer.
