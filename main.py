"""
Main Entry Point -- Insurance Health Claims RAG Pipeline Orchestrator.

Usage:
    python main.py extract   -- Extract text, images, and annotations from PDF
    python main.py embed     -- Generate page-level embeddings
    python main.py chat      -- Start interactive chat
    python main.py all       -- Run extract -> embed -> chat
    python main.py retrieve "your query here"  -- One-off retrieval test
"""

import sys
import logging
import time

from src.config import ensure_directories, GOOGLE_API_KEY


def _setup_logging():
    """Configure logging with timestamps."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )


def _check_api_key():
    """Verify API key is configured."""
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_api_key_here":
        print("\n  [ERROR] GOOGLE_API_KEY not configured!")
        print("  Set it in .env file: GOOGLE_API_KEY=your_actual_key")
        print("  Get a key at: https://aistudio.google.com/apikey\n")
        sys.exit(1)


def cmd_extract():
    """Run PDF extraction pipeline."""
    from src.extractor import extract_pdf

    print("\n" + "=" * 60)
    print("  PHASE 1: PDF Extraction")
    print("=" * 60)

    start = time.time()
    chunks = extract_pdf()
    elapsed = time.time() - start

    print(f"\n  [OK] Extraction complete: {len(chunks)} chunks in {elapsed:.1f}s")
    return chunks


def cmd_embed(chunks=None):
    """Run embedding generation."""
    from src.embedding import generate_embeddings

    print("\n" + "=" * 60)
    print("  PHASE 2: Embedding Generation")
    print("=" * 60)

    start = time.time()
    embeddings_data = generate_embeddings(chunks)
    elapsed = time.time() - start

    total = embeddings_data["total_chunks"]
    print(f"\n  [OK] Embedding complete: {total} embeddings in {elapsed:.1f}s")
    return embeddings_data


def cmd_chat():
    """Start interactive chat."""
    from src.chat import start_chat
    start_chat()


def cmd_retrieve(query: str):
    """One-off retrieval for testing."""
    from src.retriever import Retriever

    print(f"\n  [SEARCH] Retrieving for: \"{query}\"\n")

    retriever = Retriever()
    results = retriever.retrieve(query)

    for r in results:
        print(f"  [{r['chunk_id']}] Page {r['page_number']}")
        print(f"    Scores: cosine={r['scores']['cosine']:.4f}, "
              f"bm25={r['scores']['bm25']:.4f}, "
              f"hybrid={r['scores']['hybrid']:.4f}")
        headings = r.get("metadata", {}).get("headings", [])
        if headings:
            print(f"    Headings: {', '.join(headings[:3])}")
        preview = " ".join(r["content"].split())[:200]
        print(f"    Preview: {preview}...")
        print()


def cmd_all():
    """Run the full pipeline: extract -> embed -> chat."""
    chunks = cmd_extract()
    cmd_embed(chunks)
    print("\n  [OK] Pipeline complete! Starting chat...\n")
    cmd_chat()


def main():
    """Parse command and dispatch."""
    _setup_logging()
    ensure_directories()

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    command = sys.argv[1].lower()

    if command in ("extract", "chat"):
        # extract doesn't need API key check upfront (useful for testing PyMuPDF only)
        # chat needs it
        if command == "chat":
            _check_api_key()
    else:
        _check_api_key()

    if command == "extract":
        _check_api_key()
        cmd_extract()
    elif command == "embed":
        cmd_embed()
    elif command == "chat":
        cmd_chat()
    elif command == "retrieve":
        query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "What are the claim denial reasons?"
        cmd_retrieve(query)
    elif command == "all":
        cmd_all()
    else:
        print(f"\n  [ERROR] Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
