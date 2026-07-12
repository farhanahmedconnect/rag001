"""
Retriever Module — Hybrid BM25 + Semantic Retrieval with Gemini Re-ranking.

This module handles:
1. Loading embeddings from embeddings.json
2. Cosine similarity search (semantic retrieval)
3. BM25 keyword scoring (keyword retrieval)
4. Hybrid score fusion (weighted combination)
5. Gemini-based re-ranking of top candidates for higher answer quality
"""

import json
import math
import re
import logging
from collections import Counter

import numpy as np
from google import genai
from sentence_transformers import SentenceTransformer

from src.config import (
    GOOGLE_API_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    CHAT_MODEL,
    TOP_K,
    SIMILARITY_THRESHOLD,
    SEMANTIC_WEIGHT,
    BM25_WEIGHT,
    ENABLE_RERANKING,
    RERANK_CANDIDATES,
    EMBEDDINGS_FILE,
)

logger = logging.getLogger(__name__)


# ─── BM25 Scorer ───────────────────────────────────────────────────────────

class BM25Scorer:
    """
    Simple BM25 implementation for keyword-based retrieval.
    Catches exact term matches that semantic embeddings might miss
    (e.g., "CPT 99213", "ICD-10 J06.9", "HCADJ001").
    """

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.doc_count = len(documents)

        # Tokenize all documents
        self.doc_tokens = [self._tokenize(doc) for doc in documents]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_length = sum(self.doc_lengths) / max(self.doc_count, 1)

        # Build document frequency (DF) for each term
        self.df = Counter()
        for tokens in self.doc_tokens:
            unique_terms = set(tokens)
            for term in unique_terms:
                self.df[term] += 1

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer, lowercased."""
        text = text.lower()
        # Keep alphanumeric, hyphens, and dots (for codes like ICD-10, 99213)
        tokens = re.findall(r"[a-z0-9][\w\-\.]*[a-z0-9]|[a-z0-9]", text)
        return tokens

    def _idf(self, term: str) -> float:
        """Compute Inverse Document Frequency for a term."""
        df = self.df.get(term, 0)
        return math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query: str) -> list[float]:
        """
        Score all documents against a query.
        Returns list of BM25 scores (one per document).
        """
        query_tokens = self._tokenize(query)
        scores = []

        for i, doc_tokens in enumerate(self.doc_tokens):
            doc_len = self.doc_lengths[i]
            tf_map = Counter(doc_tokens)

            score = 0.0
            for term in query_tokens:
                tf = tf_map.get(term, 0)
                idf = self._idf(term)

                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
                score += idf * (numerator / denominator)

            scores.append(score)

        return scores


# ─── Cosine Similarity ─────────────────────────────────────────────────────

def _cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a query vector and all document vectors.
    Uses vectorized NumPy operations for efficiency.
    """
    # Normalize query
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        return np.zeros(len(doc_vecs))
    query_normalized = query_vec / query_norm

    # Normalize documents
    doc_norms = np.linalg.norm(doc_vecs, axis=1, keepdims=True)
    doc_norms = np.where(doc_norms == 0, 1, doc_norms)  # Avoid division by zero
    doc_normalized = doc_vecs / doc_norms

    # Compute similarities
    similarities = np.dot(doc_normalized, query_normalized)
    return similarities


# ─── Retriever Class ────────────────────────────────────────────────────────

class Retriever:
    """
    Hybrid retriever combining semantic (cosine) and keyword (BM25) search.
    Optionally re-ranks results using Gemini for higher quality.
    """

    def __init__(self, embeddings_data: dict | None = None):
        """
        Initialize retriever.
        
        Args:
            embeddings_data: Pre-loaded embeddings dict. If None, loads from disk.
        """
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        # Load embeddings
        if embeddings_data is None:
            embeddings_data = self._load_embeddings()

        self.embeddings_data = embeddings_data
        self.entries = embeddings_data["embeddings"]

        # Build numpy matrix of document embeddings
        self.doc_vectors = np.array(
            [entry["embedding"] for entry in self.entries],
            dtype=np.float32,
        )

        # Build BM25 index from chunk content
        self.documents = [entry["content"] for entry in self.entries]
        self.bm25 = BM25Scorer(self.documents)

        logger.info(
            f"[RETRIEVER] Loaded {len(self.entries)} chunks, "
            f"vector dim={self.doc_vectors.shape[1]}"
        )

    def _load_embeddings(self) -> dict:
        """Load embeddings from disk."""
        if not EMBEDDINGS_FILE.exists():
            raise FileNotFoundError(
                f"Embeddings file not found: {EMBEDDINGS_FILE}. "
                "Run embedding first: python main.py embed"
            )
        with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    def _embed_query(self, query: str) -> list[float]:
        """Generate embedding for a search query."""
        embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        return embedding[0].tolist()

    def _normalize_scores(self, scores: list[float]) -> list[float]:
        """Min-max normalize scores to [0, 1] range."""
        if not scores:
            return scores
        min_s = min(scores)
        max_s = max(scores)
        if max_s == min_s:
            return [0.5] * len(scores)
        return [(s - min_s) / (max_s - min_s) for s in scores]

    async def _rerank_with_gemini(
        self,
        query: str,
        candidates: list[dict],
    ) -> list[dict]:
        """
        Re-rank candidates using Gemini to assess actual relevance.
        Sends the query + candidate chunks to Gemini and asks for relevance ordering.
        """
        if not candidates:
            return candidates

        # Build the re-ranking prompt
        chunks_text = ""
        for i, cand in enumerate(candidates):
            # Truncate content for the prompt (avoid token limits)
            content_preview = cand["content"][:500]
            chunks_text += f"\n[Chunk {i}] (Page {cand['page_number']}, ID: {cand['chunk_id']}):\n{content_preview}\n"

        rerank_prompt = f"""Given the user query and the following document chunks, rank them by relevance.
Return ONLY a JSON array of chunk indices (0-based) ordered from most relevant to least relevant.
Include only chunks that are genuinely relevant to the query.

Query: "{query}"

Chunks:
{chunks_text}

Return format: [2, 0, 4, 1]  (just the array of indices, most relevant first)
"""

        try:
            response = await self.client.aio.models.generate_content(
                model=CHAT_MODEL,
                contents=rerank_prompt,
            )

            response_text = response.text.strip()
            # Handle markdown code fences
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1])

            ranked_indices = json.loads(response_text)

            if isinstance(ranked_indices, list):
                reranked = []
                for idx in ranked_indices:
                    if isinstance(idx, int) and 0 <= idx < len(candidates):
                        reranked.append(candidates[idx])
                return reranked if reranked else candidates

        except Exception as e:
            logger.warning(f"  [RERANK] Gemini re-ranking failed: {e}. Using original order.")

        return candidates

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        use_reranking: bool | None = None,
    ) -> list[dict]:
        """
        Retrieve the most relevant chunks for a query using hybrid search.
        
        Args:
            query: The search query string
            top_k: Number of results to return (default: config.TOP_K)
            use_reranking: Whether to use Gemini re-ranking (default: config.ENABLE_RERANKING)
        
        Returns:
            List of result dicts with chunk info and scores.
        """
        if top_k is None:
            top_k = TOP_K
        if use_reranking is None:
            use_reranking = ENABLE_RERANKING

        logger.info(f"[RETRIEVER] Query: \"{query[:80]}...\"")

        # 1. Semantic search — cosine similarity
        query_vector = np.array(self._embed_query(query), dtype=np.float32)
        cosine_scores = _cosine_similarity(query_vector, self.doc_vectors).tolist()
        cosine_normalized = self._normalize_scores(cosine_scores)

        # 2. BM25 keyword search
        bm25_scores = self.bm25.score(query)
        bm25_normalized = self._normalize_scores(bm25_scores)

        # 3. Hybrid fusion — weighted combination
        hybrid_scores = []
        for i in range(len(self.entries)):
            hybrid = (
                SEMANTIC_WEIGHT * cosine_normalized[i]
                + BM25_WEIGHT * bm25_normalized[i]
            )
            hybrid_scores.append(hybrid)

        # 4. Sort by hybrid score, get top candidates
        num_candidates = RERANK_CANDIDATES if use_reranking else top_k
        indexed_scores = list(enumerate(hybrid_scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = indexed_scores[:num_candidates]

        # 5. Filter by similarity threshold
        candidates = []
        for idx, score in top_indices:
            if cosine_scores[idx] < SIMILARITY_THRESHOLD:
                continue

            entry = self.entries[idx]
            result = {
                "chunk_id": entry["chunk_id"],
                "page_number": entry["page_number"],
                "content": entry["content"],
                "metadata": entry["metadata"],
                "has_image_annotations": entry.get("has_image_annotations", False),
                "scores": {
                    "hybrid": round(score, 4),
                    "cosine": round(cosine_scores[idx], 4),
                    "bm25": round(bm25_scores[idx], 4),
                    "cosine_normalized": round(cosine_normalized[idx], 4),
                    "bm25_normalized": round(bm25_normalized[idx], 4),
                },
            }
            candidates.append(result)

        # 6. Re-rank with Gemini (if enabled)
        if use_reranking and len(candidates) > top_k:
            import asyncio
            logger.info(f"  [RERANK] Re-ranking {len(candidates)} candidates with {CHAT_MODEL}...")
            candidates = asyncio.run(self._rerank_with_gemini(query, candidates))

        # 7. Return top_k results
        results = candidates[:top_k]

        logger.info(
            f"[RETRIEVER] Returning {len(results)} results "
            f"(best cosine={results[0]['scores']['cosine']:.4f})" if results else "[RETRIEVER] No results"
        )

        return results


def retrieve(query: str, top_k: int | None = None) -> list[dict]:
    """Convenience function for one-off retrieval."""
    retriever = Retriever()
    return retriever.retrieve(query, top_k=top_k)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What are the claim denial reasons?"
    results = retrieve(query)

    for r in results:
        print(f"\n{'='*60}")
        print(f"Chunk: {r['chunk_id']} | Page: {r['page_number']}")
        print(f"Scores: hybrid={r['scores']['hybrid']}, cosine={r['scores']['cosine']}, bm25={r['scores']['bm25']}")
        print(f"Content preview: {r['content'][:200]}...")
