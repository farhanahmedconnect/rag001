"""
Configuration for the Insurance Health Claims RAG Pipeline.
All tunable parameters are centralized here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ─── Load Environment Variables ─────────────────────────────────────────────
load_dotenv()

# ─── Project Paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
EXTRACTED_DIR = OUTPUT_DIR / "extracted"
IMAGES_DIR = OUTPUT_DIR / "images"
CACHE_DIR = OUTPUT_DIR / "cache"
EMBEDDINGS_FILE = OUTPUT_DIR / "embeddings.json"

# ─── Google Gemini API ──────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Model for image annotation (layout processing + text extraction)
ANNOTATION_MODEL = "gemini-3.1-flash-lite"

# Model for chat / re-ranking
CHAT_MODEL = "gemini-3.1-flash-lite"

# Model for embeddings (sentence-transformers)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Embedding vector dimension (384 for MiniLM)
EMBEDDING_DIM = 384

# ─── Extraction Settings ────────────────────────────────────────────────────
# Minimum number of vector drawing items on a page to trigger Gemini annotation.
# Pages with more than this threshold are considered to contain diagrams/flowcharts.
VECTOR_LINE_THRESHOLD = 14

# Minimum cluster size (in points) to be considered a significant diagram.
# Smaller clusters are likely decorative elements (borders, underlines).
MIN_CLUSTER_SIZE = 20

# DPI for rendering diagram regions as images for Gemini annotation.
RENDER_DPI = 200

# Maximum concurrent Gemini annotation calls (respects API rate limits).
MAX_PARALLEL_ANNOTATIONS = 5

# ─── Page Boundary Overlap ──────────────────────────────────────────────────
# Number of characters to overlap from the previous/next page for context continuity.
PAGE_OVERLAP_CHARS = 200

# ─── Embedding Settings ────────────────────────────────────────────────────
# Task type for document embeddings (RETRIEVAL_DOCUMENT for indexing)
EMBED_TASK_TYPE_DOCUMENT = "RETRIEVAL_DOCUMENT"

# Task type for query embeddings (RETRIEVAL_QUERY for search)
EMBED_TASK_TYPE_QUERY = "RETRIEVAL_QUERY"

# ─── Retrieval Settings ────────────────────────────────────────────────────
# Number of top results to retrieve
TOP_K = 5

# Minimum cosine similarity score to include in results
SIMILARITY_THRESHOLD = 0.3

# Weight for semantic (cosine) score in hybrid retrieval (0.0 to 1.0)
SEMANTIC_WEIGHT = 0.7

# Weight for BM25 keyword score in hybrid retrieval
BM25_WEIGHT = 0.3

# Whether to enable Gemini re-ranking of retrieved results
ENABLE_RERANKING = True

# Number of candidates to fetch before re-ranking (fetch more, then re-rank to TOP_K)
RERANK_CANDIDATES = 10

# ─── Annotation Prompt ─────────────────────────────────────────────────────
ANNOTATION_PROMPT = """You are a layout processor and text extractor for insurance documents.
Analyze this image from a health insurance claims document and provide a structured JSON response.

Respond ONLY with valid JSON in this exact format:
{
  "diagram_type": "<flowchart|table|form|screen|chart|infographic|other>",
  "title": "<title or heading visible in the image>",
  "text_content": "<ALL text content visible in the image, preserving structure>",
  "layout_description": "<brief description of the visual layout and arrangement>",
  "relationships": "<describe any flows, connections, hierarchies shown>",
  "data_points": ["<key data point 1>", "<key data point 2>", "..."],
  "mainframe_codes": ["<any system codes, transaction IDs, or reference numbers visible>"]
}

Be thorough in extracting ALL visible text. For mainframe screens, capture every field and value.
For flowcharts, describe the flow direction and all decision points.
For tables, capture all rows and columns."""

# ─── Chat System Prompt ────────────────────────────────────────────────────
CHAT_SYSTEM_PROMPT = """You are an expert insurance claims assistant. Your role is to answer questions
about health insurance claim rules, processes, and guidelines based ONLY on the provided context
from the health claims rules document.

Rules:
1. Answer based ONLY on the provided context. Do not make up information.
2. If the context does not contain enough information to answer, say so clearly.
3. Reference specific page numbers and sections when possible.
4. For mainframe codes or system references, quote them exactly as they appear.
5. Be precise and professional in your responses.
6. If multiple chunks are relevant, synthesize information from all of them."""

# ─── Structured Separators for Combined Content ────────────────────────────
SECTION_SEPARATOR_PAGE_TEXT = "[PAGE TEXT]"
SECTION_SEPARATOR_IMAGE_ANNOTATION = "[DIAGRAM ANNOTATION]"
SECTION_SEPARATOR_OVERLAP_BEFORE = "[CONTEXT FROM PREVIOUS PAGE]"
SECTION_SEPARATOR_OVERLAP_AFTER = "[CONTEXT FROM NEXT PAGE]"

# ─── Ensure directories exist ──────────────────────────────────────────────
def ensure_directories():
    """Create all required output directories."""
    for directory in [INPUT_DIR, OUTPUT_DIR, EXTRACTED_DIR, IMAGES_DIR, CACHE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
