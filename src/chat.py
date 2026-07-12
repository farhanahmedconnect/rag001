"""
Chat Module — Interactive CLI RAG Chat with Source Attribution.

This module:
1. Provides an interactive CLI chat loop
2. Retrieves relevant chunks for each user question
3. Sends context + question to Gemini for grounded answers
4. Displays retrieved pages/chunks alongside the answer
5. Maintains conversation history for multi-turn context
6. Exits on 'exit' or 'bye'
"""

import sys
import logging

from google import genai
from google.genai import types

from src.config import (
    GOOGLE_API_KEY,
    CHAT_MODEL,
    CHAT_SYSTEM_PROMPT,
    TOP_K,
)
from src.retriever import Retriever

logger = logging.getLogger(__name__)


# ─── Display Helpers ───────────────────────────────────────────────────────

def _print_header():
    """Print the chat welcome header."""
    print("\n" + "=" * 70)
    print("  Insurance Health Claims RAG -- Interactive Chat")
    print("=" * 70)
    print(f"  Model: {CHAT_MODEL}")
    print(f"  Retrieval: Hybrid (BM25 + Semantic) with Re-ranking")
    print(f"  Top-K: {TOP_K}")
    print("  Type 'exit' or 'bye' to quit.")
    print("=" * 70 + "\n")


def _print_retrieved_chunks(results: list[dict]):
    """Display the retrieved chunks with scores and metadata."""
    print("\n" + "-" * 50)
    print(f"  Retrieved {len(results)} chunk(s):")
    print("-" * 50)

    for i, r in enumerate(results):
        chunk_id = r["chunk_id"]
        page_num = r["page_number"]
        cosine = r["scores"]["cosine"]
        hybrid = r["scores"]["hybrid"]
        has_images = r.get("has_image_annotations", False)
        headings = r.get("metadata", {}).get("headings", [])

        # Compact heading display
        heading_str = ""
        if headings:
            heading_str = f" | Headings: {', '.join(headings[:2])}"

        image_str = " [IMG]" if has_images else ""

        # Content preview (first 150 chars, cleaned)
        content = r["content"]
        # Strip structured separators for preview
        for sep in ["[PAGE TEXT]", "[DIAGRAM ANNOTATION]", "[CONTEXT FROM PREVIOUS PAGE]", "[CONTEXT FROM NEXT PAGE]"]:
            content = content.replace(sep, "")
        preview = " ".join(content.split())[:150]

        print(
            f"\n  [{i + 1}] {chunk_id} | Page {page_num} | "
            f"cosine={cosine:.3f} hybrid={hybrid:.3f}{image_str}{heading_str}"
        )
        print(f"      {preview}...")

    print("\n" + "-" * 50)


def _build_context(results: list[dict]) -> str:
    """Build the context string from retrieved chunks."""
    context_parts = []
    for r in results:
        chunk_id = r["chunk_id"]
        page_num = r["page_number"]
        content = r["content"]
        context_parts.append(
            f"[Source: {chunk_id}, Page {page_num}]\n{content}"
        )
    return "\n\n---\n\n".join(context_parts)


# ─── Chat Loop ─────────────────────────────────────────────────────────────

def start_chat():
    """
    Start the interactive CLI chat loop.
    
    Features:
    - Retrieves relevant chunks per question
    - Shows retrieved pages/chunks with scores
    - Sends grounded context to Gemini
    - Maintains conversation history
    - Exits on 'exit' or 'bye'
    """
    _print_header()

    # Initialize components
    try:
        retriever = Retriever()
    except FileNotFoundError as e:
        print(f"\n  [ERROR] Error: {e}")
        print("  Run the pipeline first: python main.py all")
        sys.exit(1)

    client = genai.Client(api_key=GOOGLE_API_KEY)

    # Conversation history for multi-turn context
    conversation_history: list[dict] = []

    while True:
        # Get user input
        try:
            user_input = input("\n  You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Goodbye!")
            break

        # Check for exit commands
        if not user_input:
            continue
        if user_input.lower() in ("exit", "bye", "quit", "q"):
            print("\n  Goodbye!")
            break

        # ── Step 1: Retrieve relevant chunks ──────────────────────────
        results = retriever.retrieve(user_input)

        if not results:
            print("\n  [WARNING] No relevant chunks found for your query.")
            print("  Try rephrasing your question or using different keywords.")
            continue

        # ── Step 2: Display retrieved chunks ──────────────────────────
        _print_retrieved_chunks(results)

        # ── Step 3: Build context and generate answer ─────────────────
        context = _build_context(results)

        # Build the prompt with context and conversation history
        prompt_parts = [CHAT_SYSTEM_PROMPT + "\n"]

        # Add conversation history (last 3 turns for context)
        if conversation_history:
            prompt_parts.append("\n--- Previous Conversation ---")
            for turn in conversation_history[-3:]:
                prompt_parts.append(f"User: {turn['question']}")
                prompt_parts.append(f"Assistant: {turn['answer'][:300]}...")

        prompt_parts.append(f"\n--- Retrieved Context ---\n{context}")
        prompt_parts.append(f"\n--- Current Question ---\nUser: {user_input}")
        prompt_parts.append(
            "\nProvide a detailed, accurate answer based on the context above. "
            "Reference specific page numbers and chunk IDs when possible."
        )

        full_prompt = "\n".join(prompt_parts)

        try:
            response = client.models.generate_content(
                model=CHAT_MODEL,
                contents=full_prompt,
            )

            answer = response.text

            # Display the answer
            print(f"\n  Assistant:\n")
            # Indent the answer for readability
            for line in answer.split("\n"):
                print(f"      {line}")

            # Display source attribution
            source_pages = sorted(set(r["page_number"] for r in results))
            source_chunks = [r["chunk_id"] for r in results]
            print(f"\n  Sources: Pages {source_pages} | Chunks: {', '.join(source_chunks)}")

            # Save to conversation history
            conversation_history.append({
                "question": user_input,
                "answer": answer,
                "sources": source_chunks,
            })

        except Exception as e:
            print(f"\n  [ERROR] Error generating response: {e}")
            logger.error(f"Chat error: {e}", exc_info=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    start_chat()
