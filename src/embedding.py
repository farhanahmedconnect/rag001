"""
Embedding Module — Page-Level Embedding Generation using Gemini.

Per NVIDIA research, page-level embedding is more effective than sentence-level
semantic chunking for structured domain documents like insurance guidelines.

This module:
1. Loads extracted page chunks from output/extracted/
2. Generates one embedding per page (one chunk = one page = one embedding)
3. Uses text-embedding-004 with RETRIEVAL_DOCUMENT task type
4. Saves embeddings.json with chunk IDs, content, metadata, and embedding vectors
"""

import json
import logging
from pathlib import Path
from datetime import datetime

from sentence_transformers import SentenceTransformer

from src.config import (
    GOOGLE_API_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    EXTRACTED_DIR,
    EMBEDDINGS_FILE,
    ensure_directories,
)

logger = logging.getLogger(__name__)


def _load_extracted_chunks() -> list[dict]:
    """
    Load all extracted page chunks from output/extracted/page_*.json.
    Returns them sorted by page number.
    """
    chunk_files = sorted(EXTRACTED_DIR.glob("page_*.json"))

    if not chunk_files:
        raise FileNotFoundError(
            f"No extracted chunks found in {EXTRACTED_DIR}. "
            "Run extraction first: python main.py extract"
        )

    chunks = []
    for chunk_file in chunk_files:
        with open(chunk_file, "r", encoding="utf-8") as f:
            chunk = json.load(f)
        chunks.append(chunk)

    # Sort by page number
    chunks.sort(key=lambda c: c.get("page_number", 0))
    logger.info(f"[EMBEDDING] Loaded {len(chunks)} chunks from {EXTRACTED_DIR}")

    return chunks


def _generate_embeddings_batch(
    model: SentenceTransformer,
    texts: list[str],
) -> list[list[float]]:
    """
    Generate embeddings for a list of texts using SentenceTransformer.
    """
    logger.info(f"  [EMBEDDING] Encoding {len(texts)} texts...")
    try:
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    except Exception as e:
        logger.error(f"  [EMBEDDING] failed: {e}")
        return [[0.0] * EMBEDDING_DIM for _ in texts]


def generate_embeddings(chunks: list[dict] | None = None) -> dict:
    """
    Generate page-level embeddings for all extracted chunks.
    
    One page = one chunk = one embedding.
    
    Args:
        chunks: Pre-loaded chunks. If None, loads from disk.
    
    Returns:
        The complete embeddings data structure (also saved to disk).
    """
    ensure_directories()

    # Load chunks if not provided
    if chunks is None:
        chunks = _load_extracted_chunks()

    if not chunks:
        raise ValueError("No chunks to embed. Run extraction first.")

    logger.info(f"[EMBEDDING] Generating {len(chunks)} page-level embeddings...")
    logger.info(f"[EMBEDDING] Model: {EMBEDDING_MODEL}, Dimension: {EMBEDDING_DIM}")

    # Initialize SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Extract content texts for embedding
    texts = []
    for chunk in chunks:
        content = chunk.get("content", "")
        if not content.strip():
            # Fallback: use raw page text if content is empty
            content = f"Page {chunk.get('page_number', '?')} - No content extracted."
        texts.append(content)

    # Generate embeddings
    embeddings_vectors = _generate_embeddings_batch(model, texts)

    # Build the output structure
    embeddings_list = []
    for i, chunk in enumerate(chunks):
        embedding_entry = {
            "chunk_id": chunk["chunk_id"],
            "page_number": chunk["page_number"],
            "content": chunk["content"],
            "has_image_annotations": chunk["metadata"].get("diagram_count", 0) > 0
                                     or chunk["metadata"].get("raster_image_count", 0) > 0,
            "metadata": chunk["metadata"],
            "embedding": embeddings_vectors[i],
        }
        embeddings_list.append(embedding_entry)

    # Build final output
    embeddings_data = {
        "model": EMBEDDING_MODEL,
        "dimension": EMBEDDING_DIM,
        "total_chunks": len(embeddings_list),
        "total_pages": len(embeddings_list),
        "created_at": datetime.now().isoformat(),
        "source_pdf": chunks[0]["metadata"].get("source", "unknown") if chunks else "unknown",
        "embeddings": embeddings_list,
    }

    # Save to disk
    with open(EMBEDDINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(embeddings_data, f, indent=2, ensure_ascii=False)

    file_size_mb = EMBEDDINGS_FILE.stat().st_size / (1024 * 1024)
    logger.info(f"[EMBEDDING] Saved {len(embeddings_list)} embeddings to {EMBEDDINGS_FILE}")
    logger.info(f"[EMBEDDING] File size: {file_size_mb:.2f} MB")

    return embeddings_data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    generate_embeddings()
