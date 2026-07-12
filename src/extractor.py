"""
PDF Extractor Module — Text & Image Extraction using PyMuPDF + Gemini Annotation.

This module handles:
1. Page-wise text extraction from PDF
2. Raster image extraction
3. Vector diagram detection (drawings with >VECTOR_LINE_THRESHOLD lines)
4. Parallel Gemini annotation of detected diagrams/images
5. Structured JSON output per page with chunk IDs
6. Annotation caching to avoid re-processing
7. Image deduplication via content hash
8. Page boundary overlap for context continuity
"""

import json
import hashlib
import asyncio
import logging
from pathlib import Path
from datetime import datetime

import pymupdf
from google import genai
from google.genai import types

from src.config import (
    GOOGLE_API_KEY,
    ANNOTATION_MODEL,
    ANNOTATION_PROMPT,
    VECTOR_LINE_THRESHOLD,
    MIN_CLUSTER_SIZE,
    RENDER_DPI,
    MAX_PARALLEL_ANNOTATIONS,
    PAGE_OVERLAP_CHARS,
    INPUT_DIR,
    EXTRACTED_DIR,
    IMAGES_DIR,
    CACHE_DIR,
    SECTION_SEPARATOR_PAGE_TEXT,
    SECTION_SEPARATOR_IMAGE_ANNOTATION,
    SECTION_SEPARATOR_OVERLAP_BEFORE,
    SECTION_SEPARATOR_OVERLAP_AFTER,
    ensure_directories,
)

logger = logging.getLogger(__name__)


# ─── Annotation Cache ──────────────────────────────────────────────────────

def _get_cache_path(image_hash: str) -> Path:
    """Return the cache file path for a given image hash."""
    return CACHE_DIR / f"{image_hash}.json"


def _load_cached_annotation(image_hash: str) -> dict | None:
    """Load annotation from cache if it exists."""
    cache_path = _get_cache_path(image_hash)
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            logger.info(f"  [CACHE HIT] Loaded annotation for hash {image_hash[:12]}...")
            return cached
        except (json.JSONDecodeError, IOError):
            return None
    return None


def _save_annotation_cache(image_hash: str, annotation: dict):
    """Save annotation to cache."""
    cache_path = _get_cache_path(image_hash)
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(annotation, f, indent=2, ensure_ascii=False)
    except IOError as e:
        logger.warning(f"  [CACHE] Failed to save cache: {e}")


def _compute_image_hash(image_bytes: bytes) -> str:
    """Compute SHA-256 hash of image bytes for deduplication and caching."""
    return hashlib.sha256(image_bytes).hexdigest()


# ─── Vector Drawing Analysis ───────────────────────────────────────────────

def _count_meaningful_drawings(page) -> int:
    """
    Count meaningful vector drawing items on a page, filtering out decorative elements.
    
    Filters out:
    - Simple short horizontal/vertical lines (likely underlines, borders)
    - Very thin lines (width < 0.5) that are typically separators
    """
    drawings = page.get_drawings()
    meaningful_count = 0

    for path in drawings:
        for item in path.get("items", []):
            item_type = item[0]

            if item_type == "l":  # Line segment
                p1, p2 = item[1], item[2]
                # Calculate line length
                length = ((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2) ** 0.5
                # Skip very short lines (< 5 points ≈ decorative)
                if length < 5:
                    continue
                # Check if it's a simple horizontal/vertical line (border/underline)
                is_horizontal = abs(p2.y - p1.y) < 1
                is_vertical = abs(p2.x - p1.x) < 1
                # Only skip if it's a long straight border-like line
                if (is_horizontal or is_vertical) and length > 200:
                    continue
                meaningful_count += 1

            elif item_type in ("c", "qu"):  # Bezier curve or quad — always meaningful
                meaningful_count += 1

            elif item_type == "re":  # Rectangle
                rect = item[1]
                # Skip very thin rectangles (likely borders/rules)
                if isinstance(rect, pymupdf.Rect):
                    if rect.width < 3 or rect.height < 3:
                        continue
                meaningful_count += 1

    return meaningful_count


def _extract_diagram_regions(page) -> list[pymupdf.Rect]:
    """
    Use cluster_drawings() to find diagram regions on a page.
    Filters out clusters smaller than MIN_CLUSTER_SIZE.
    """
    try:
        clusters = page.cluster_drawings()
    except AttributeError:
        # Fallback if cluster_drawings not available in older PyMuPDF versions
        drawings = page.get_drawings()
        if not drawings:
            return []
        # Manual clustering: use the union of all drawing rects
        all_rect = pymupdf.Rect()
        for d in drawings:
            all_rect |= d.get("rect", pymupdf.Rect())
        return [all_rect] if not all_rect.is_empty else []

    # Filter significant clusters
    significant = []
    for rect in clusters:
        if isinstance(rect, pymupdf.Rect):
            if rect.width >= MIN_CLUSTER_SIZE and rect.height >= MIN_CLUSTER_SIZE:
                significant.append(rect)

    return significant


# ─── Image Extraction ──────────────────────────────────────────────────────

def _extract_raster_images(doc, page, page_num: int, seen_hashes: set) -> list[dict]:
    """
    Extract embedded raster images from a page.
    Returns list of image info dicts. Deduplicates via content hash.
    """
    images_info = []
    image_list = page.get_images(full=True)

    for img_index, img in enumerate(image_list):
        xref = img[0]
        try:
            base_image = doc.extract_image(xref)
        except Exception as e:
            logger.warning(f"  [PAGE {page_num + 1}] Failed to extract image xref={xref}: {e}")
            continue

        image_bytes = base_image["image"]
        image_ext = base_image.get("ext", "png")
        image_hash = _compute_image_hash(image_bytes)

        # Deduplication: skip if we've seen this exact image before
        if image_hash in seen_hashes:
            logger.info(f"  [PAGE {page_num + 1}] Skipping duplicate image (hash={image_hash[:12]}...)")
            continue
        seen_hashes.add(image_hash)

        # Save image to disk
        image_filename = f"page_{page_num + 1}_img_{img_index}.{image_ext}"
        image_path = IMAGES_DIR / image_filename
        with open(image_path, "wb") as f:
            f.write(image_bytes)

        images_info.append({
            "image_path": str(image_path),
            "image_hash": image_hash,
            "type": "raster",
            "xref": xref,
            "needs_annotation": True,
        })

    return images_info


def _render_diagram_regions(page, page_num: int, regions: list[pymupdf.Rect], seen_hashes: set) -> list[dict]:
    """
    Render vector diagram regions as high-res images.
    Returns list of image info dicts. Deduplicates via content hash.
    """
    images_info = []

    for i, rect in enumerate(regions):
        # Add small padding and clip to page bounds
        padded = rect + (-2, -2, 2, 2)
        padded = padded & page.rect

        # Render at high DPI
        zoom = RENDER_DPI / 72.0
        matrix = pymupdf.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, clip=padded)
        image_bytes = pix.tobytes("png")

        image_hash = _compute_image_hash(image_bytes)

        # Deduplication
        if image_hash in seen_hashes:
            logger.info(f"  [PAGE {page_num + 1}] Skipping duplicate diagram region (hash={image_hash[:12]}...)")
            continue
        seen_hashes.add(image_hash)

        # Save rendered diagram
        image_filename = f"page_{page_num + 1}_diagram_{i}.png"
        image_path = IMAGES_DIR / image_filename
        pix.save(str(image_path))

        images_info.append({
            "image_path": str(image_path),
            "image_hash": image_hash,
            "type": "vector_diagram",
            "region": [rect.x0, rect.y0, rect.x1, rect.y1],
            "needs_annotation": True,
        })

    return images_info


# ─── Heading Detection ─────────────────────────────────────────────────────

def _detect_headings(page) -> list[str]:
    """
    Detect headings on a page by analyzing font sizes.
    Returns list of heading strings (text with font size significantly larger than body).
    """
    headings = []
    try:
        blocks = page.get_text("dict")["blocks"]
    except Exception:
        return headings

    # Collect all font sizes to find the median (body text size)
    all_sizes = []
    for block in blocks:
        if block.get("type") != 0:  # Skip image blocks
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                size = span.get("size", 0)
                if size > 0:
                    all_sizes.append(size)

    if not all_sizes:
        return headings

    # Body text is typically the most common font size
    median_size = sorted(all_sizes)[len(all_sizes) // 2]
    heading_threshold = median_size * 1.2  # 20% larger than body = heading

    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if span.get("size", 0) >= heading_threshold:
                    text = span.get("text", "").strip()
                    if text and len(text) > 2:  # Skip single chars
                        headings.append(text)

    return headings


# ─── Gemini Annotation (Parallel) ──────────────────────────────────────────

async def _annotate_image_async(
    client: genai.Client,
    image_info: dict,
    semaphore: asyncio.Semaphore,
) -> dict:
    """
    Send a single image to Gemini for layout annotation + text extraction.
    Uses async API with semaphore for rate limiting.
    """
    image_path = image_info["image_path"]
    image_hash = image_info["image_hash"]

    # Check cache first
    cached = _load_cached_annotation(image_hash)
    if cached:
        image_info["annotation"] = cached
        image_info["needs_annotation"] = False
        return image_info

    async with semaphore:
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()

            mime_type = "image/png"
            if image_path.lower().endswith((".jpg", ".jpeg")):
                mime_type = "image/jpeg"

            response = await client.aio.models.generate_content(
                model=ANNOTATION_MODEL,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    ANNOTATION_PROMPT,
                ],
            )

            # Parse JSON response
            response_text = response.text.strip()
            # Handle markdown code fences around JSON
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1])

            try:
                annotation = json.loads(response_text)
            except json.JSONDecodeError:
                # If JSON parsing fails, use raw text as annotation
                annotation = {
                    "diagram_type": "unknown",
                    "title": "",
                    "text_content": response_text,
                    "layout_description": "",
                    "relationships": "",
                    "data_points": [],
                    "mainframe_codes": [],
                }

            # Cache the annotation
            _save_annotation_cache(image_hash, annotation)

            image_info["annotation"] = annotation
            image_info["needs_annotation"] = False
            logger.info(f"  [ANNOTATED] {Path(image_path).name} → {annotation.get('diagram_type', 'unknown')}")

        except Exception as e:
            logger.error(f"  [ANNOTATION FAILED] {Path(image_path).name}: {e}")
            image_info["annotation"] = {
                "diagram_type": "error",
                "title": "",
                "text_content": f"Annotation failed: {str(e)}",
                "layout_description": "",
                "relationships": "",
                "data_points": [],
                "mainframe_codes": [],
            }
            image_info["needs_annotation"] = False

    return image_info


async def _annotate_all_images(
    client: genai.Client,
    images: list[dict],
) -> list[dict]:
    """Annotate all images in parallel with concurrency control."""
    if not images:
        return images

    semaphore = asyncio.Semaphore(MAX_PARALLEL_ANNOTATIONS)
    tasks = [_annotate_image_async(client, img, semaphore) for img in images]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    annotated = []
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"  [ANNOTATION ERROR] {result}")
        else:
            annotated.append(result)

    return annotated


# ─── Main Extraction Pipeline ──────────────────────────────────────────────

def _format_annotation_text(annotation: dict) -> str:
    """Convert a structured annotation dict to readable text for embedding."""
    parts = []

    if annotation.get("title"):
        parts.append(f"Title: {annotation['title']}")

    if annotation.get("diagram_type"):
        parts.append(f"Type: {annotation['diagram_type']}")

    if annotation.get("layout_description"):
        parts.append(f"Layout: {annotation['layout_description']}")

    if annotation.get("text_content"):
        parts.append(f"Content: {annotation['text_content']}")

    if annotation.get("relationships"):
        parts.append(f"Relationships: {annotation['relationships']}")

    if annotation.get("data_points"):
        points = annotation["data_points"]
        if isinstance(points, list):
            parts.append(f"Data Points: {'; '.join(str(p) for p in points)}")

    if annotation.get("mainframe_codes"):
        codes = annotation["mainframe_codes"]
        if isinstance(codes, list):
            parts.append(f"Mainframe Codes: {', '.join(str(c) for c in codes)}")

    return "\n".join(parts)


def _build_chunk_content(
    page_text: str,
    image_annotations: list[dict],
    prev_page_text: str | None,
    next_page_text: str | None,
) -> str:
    """
    Build the final chunk content with structured separators.
    Combines page text + image annotations + page boundary overlaps.
    """
    sections = []

    # Page boundary overlap: context from previous page
    if prev_page_text and len(prev_page_text) > 0:
        overlap = prev_page_text[-PAGE_OVERLAP_CHARS:]
        sections.append(f"{SECTION_SEPARATOR_OVERLAP_BEFORE}\n{overlap}")

    # Main page text
    sections.append(f"{SECTION_SEPARATOR_PAGE_TEXT}\n{page_text}")

    # Image/diagram annotations
    for ann in image_annotations:
        ann_text = _format_annotation_text(ann)
        if ann_text.strip():
            sections.append(f"{SECTION_SEPARATOR_IMAGE_ANNOTATION}\n{ann_text}")

    # Page boundary overlap: context from next page
    if next_page_text and len(next_page_text) > 0:
        overlap = next_page_text[:PAGE_OVERLAP_CHARS]
        sections.append(f"{SECTION_SEPARATOR_OVERLAP_AFTER}\n{overlap}")

    return "\n\n".join(sections)


def extract_pdf(pdf_path: str | Path | None = None) -> list[dict]:
    """
    Main extraction function.
    
    Extracts text, images, and diagrams from a PDF file.
    Sends diagrams to Gemini for annotation.
    Produces one chunk per page with structured content.
    
    Args:
        pdf_path: Path to the PDF file. If None, uses the first PDF in INPUT_DIR.
    
    Returns:
        List of chunk dicts, one per page.
    """
    ensure_directories()

    # Find PDF
    if pdf_path is None:
        pdfs = list(INPUT_DIR.glob("*.pdf"))
        if not pdfs:
            raise FileNotFoundError(f"No PDF files found in {INPUT_DIR}")
        pdf_path = pdfs[0]
    pdf_path = Path(pdf_path)

    logger.info(f"[EXTRACTOR] Opening PDF: {pdf_path.name}")

    # Initialize Gemini client
    client = genai.Client(api_key=GOOGLE_API_KEY)

    # Open document
    doc = pymupdf.open(str(pdf_path))
    total_pages = len(doc)
    logger.info(f"[EXTRACTOR] Total pages: {total_pages}")

    # Global deduplication set (tracks image hashes across all pages)
    seen_hashes: set[str] = set()

    # ── Phase 1: Extract text and detect images per page ────────────────
    page_data = []
    all_images_to_annotate = []

    for page_num in range(total_pages):
        page = doc[page_num]
        logger.info(f"\n[PAGE {page_num + 1}/{total_pages}] Processing...")

        # 1. Extract text
        page_text = page.get_text("text", sort=True)
        logger.info(f"  Text: {len(page_text)} chars")

        # 2. Detect headings
        headings = _detect_headings(page)
        if headings:
            logger.info(f"  Headings: {headings}")

        # 3. Extract raster images
        raster_images = _extract_raster_images(doc, page, page_num, seen_hashes)
        logger.info(f"  Raster images: {len(raster_images)}")

        # 4. Detect vector diagrams
        vector_count = _count_meaningful_drawings(page)
        logger.info(f"  Meaningful vector drawings: {vector_count}")

        diagram_images = []
        if vector_count > VECTOR_LINE_THRESHOLD:
            regions = _extract_diagram_regions(page)
            diagram_images = _render_diagram_regions(page, page_num, regions, seen_hashes)
            logger.info(f"  Diagram regions extracted: {len(diagram_images)}")

        # Combine all images that need annotation
        all_page_images = raster_images + diagram_images
        for img in all_page_images:
            img["page_number"] = page_num + 1

        all_images_to_annotate.extend(all_page_images)

        page_data.append({
            "page_number": page_num + 1,
            "text": page_text,
            "headings": headings,
            "raster_images": raster_images,
            "diagram_images": diagram_images,
            "vector_line_count": vector_count,
            "has_diagrams": (vector_count > VECTOR_LINE_THRESHOLD) or (len(raster_images) > 0),
        })

    doc.close()

    # ── Phase 2: Parallel Gemini annotation of all images ───────────────
    logger.info(f"\n[ANNOTATION] Sending {len(all_images_to_annotate)} images to Gemini ({ANNOTATION_MODEL})...")

    if all_images_to_annotate:
        annotated_images = asyncio.run(
            _annotate_all_images(client, all_images_to_annotate)
        )
        # Map annotated images back to their pages
        annotated_by_page: dict[int, list[dict]] = {}
        for img in annotated_images:
            pg = img["page_number"]
            annotated_by_page.setdefault(pg, []).append(img)
    else:
        annotated_by_page = {}

    # ── Phase 3: Build chunks with structured content ──────────────────
    logger.info("\n[CHUNKS] Building page-level chunks...")

    # Collect all page texts for boundary overlap
    all_page_texts = [pd["text"] for pd in page_data]

    chunks = []
    for i, pd in enumerate(page_data):
        page_num = pd["page_number"]
        chunk_id = f"chunk_{page_num:03d}"

        # Get annotations for this page
        page_annotations = [
            img.get("annotation", {})
            for img in annotated_by_page.get(page_num, [])
            if img.get("annotation")
        ]

        # Get overlap texts
        prev_text = all_page_texts[i - 1] if i > 0 else None
        next_text = all_page_texts[i + 1] if i < len(all_page_texts) - 1 else None

        # Build combined content
        content = _build_chunk_content(
            page_text=pd["text"],
            image_annotations=page_annotations,
            prev_page_text=prev_text,
            next_page_text=next_text,
        )

        # Build image info for the chunk (without the raw bytes/large data)
        images_metadata = []
        for img in annotated_by_page.get(page_num, []):
            images_metadata.append({
                "type": img["type"],
                "annotation": img.get("annotation", {}),
            })

        chunk = {
            "chunk_id": chunk_id,
            "page_number": page_num,
            "content": content,
            "metadata": {
                "source": pdf_path.name,
                "page": page_num,
                "total_pages": total_pages,
                "headings": pd["headings"],
                "has_diagrams": pd["has_diagrams"],
                "vector_line_count": pd["vector_line_count"],
                "raster_image_count": len(pd["raster_images"]),
                "diagram_count": len(pd["diagram_images"]),
                "content_length": len(content),
            },
            "images": images_metadata,
        }

        chunks.append(chunk)

        # Save individual page JSON
        page_json_path = EXTRACTED_DIR / f"page_{page_num}.json"
        with open(page_json_path, "w", encoding="utf-8") as f:
            json.dump(chunk, f, indent=2, ensure_ascii=False)

        logger.info(
            f"  {chunk_id}: page {page_num}, "
            f"{len(content)} chars, "
            f"{len(images_metadata)} images, "
            f"headings={pd['headings'][:2]}..."
        )

    # Save summary manifest
    manifest = {
        "source_pdf": pdf_path.name,
        "total_pages": total_pages,
        "total_chunks": len(chunks),
        "total_images_annotated": len(all_images_to_annotate),
        "extracted_at": datetime.now().isoformat(),
        "chunks": [
            {
                "chunk_id": c["chunk_id"],
                "page_number": c["page_number"],
                "content_length": c["metadata"]["content_length"],
                "has_diagrams": c["metadata"]["has_diagrams"],
            }
            for c in chunks
        ],
    }
    manifest_path = EXTRACTED_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info(f"\n[EXTRACTOR] Done. {len(chunks)} chunks extracted to {EXTRACTED_DIR}")
    return chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    extract_pdf()
