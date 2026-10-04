"""
rag/store.py — chromadb wrapper

Local, embedded vector DB. Every chunk carries full source metadata
(ticker, doc_type, published_date, source, url) so citations are always
possible -- retrieval never returns bare text with no source.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions

from rag.ingest import Document

logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).parent / "chroma_store"
_COLLECTION_NAME = "research_documents"
_CHUNK_CHARS = 2000  # roughly 500 tokens, simple v1 chunking

_collection = None  # lazily created -- avoids network/model-load at import time


def _get_collection():
    """Lazily create the chromadb client/collection on first use."""
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(_DB_PATH))
        embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        _collection = client.get_or_create_collection(name=_COLLECTION_NAME, embedding_function=embedder)
    return _collection


def _chunk_text(text: str, chunk_size: int = _CHUNK_CHARS) -> list[str]:
    """Simple paragraph-aware chunking -- no need for anything fancier in v1."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs and text.strip():
        paragraphs = [text.strip()]

    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > chunk_size:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


def add_documents(docs: list[Document]) -> int:
    """Chunk, embed, and store documents with full metadata on every chunk."""
    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict] = []

    for doc in docs:
        chunks = _chunk_text(doc.content)
        for i, chunk in enumerate(chunks):
            doc_key = doc.url or doc.title
            doc_id = f"{doc.ticker}:{doc.doc_type}:{doc.published_date.isoformat()}:{hash(doc_key)}:{i}"
            ids.append(doc_id)
            texts.append(chunk)
            metadatas.append({
                "ticker": doc.ticker,
                "doc_type": doc.doc_type,
                "title": doc.title,
                "source": doc.source,
                "url": doc.url or "",
                "published_date": doc.published_date.isoformat(),
                "published_date_ordinal": doc.published_date.toordinal(),  # chroma $gte needs a number, not a string
                "ingested_at": doc.ingested_at.isoformat(),
            })

    if not ids:
        return 0

    _get_collection().upsert(ids=ids, documents=texts, metadatas=metadatas)
    logger.info(f"Stored {len(ids)} chunks from {len(docs)} documents")
    return len(ids)


def query(
    ticker: str,
    query_text: str,
    k: int = 8,
    max_age_days: Optional[int] = None,
) -> list[dict]:
    """
    Query chunks for a ticker, optionally filtered by max age.

    Returns [{"text": ..., "metadata": {...}, "distance": ...}, ...] --
    never bare text with no source attached.
    """
    if max_age_days is not None:
        cutoff_ordinal = (date.today() - timedelta(days=max_age_days)).toordinal()
        where: dict = {"$and": [{"ticker": ticker}, {"published_date_ordinal": {"$gte": cutoff_ordinal}}]}
    else:
        where = {"ticker": ticker}

    try:
        result = _get_collection().query(query_texts=[query_text], n_results=k, where=where)
    except Exception as exc:
        logger.error(f"[{ticker}] Vector store query failed: {exc}")
        return []

    hits: list[dict] = []
    docs_list = (result.get("documents") or [[]])[0]
    metas_list = (result.get("metadatas") or [[]])[0]
    dists_list = (result.get("distances") or [[]])[0]
    for text, meta, dist in zip(docs_list, metas_list, dists_list):
        hits.append({"text": text, "metadata": meta, "distance": dist})
    return hits
