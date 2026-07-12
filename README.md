# RAG001 — PDF Parser & RAG System Template

A robust, production-ready Retrieval-Augmented Generation (RAG) system template built to process complex PDF documents containing structured text, embedded flowcharts, and user interface screenshots.

## Features
- **Page-Level Chunking**: Processes the input document page-by-page (optimized for document layout mapping and structural references).
- **Asynchronous Layout Annotation**: Automatically detects and extracts raster images and vector diagrams using PyMuPDF and annotates them asynchronously using the Gemini API.
- **Hybrid Retrieval**: Combines BM25 keyword scoring (perfect for exact matching of technical terms, CPT/ICD codes, or system identifiers) with Sentence-Transformers semantic embeddings.
- **Gemini Re-ranking**: Uses a flash-lite model to dynamically re-rank top candidates for higher response accuracy.
- **Interactive CLI**: Chat interface displaying citation sources, page references, and similarity scores.

## Project Structure
```text
├── input/
│   └── document.pdf             # Input reference PDF
├── src/
│   ├── __init__.py
│   ├── config.py                # Pipeline and model configurations
│   ├── extractor.py             # PyMuPDF parser + Gemini visual annotations
│   ├── embedding.py             # Local SentenceTransformer embeddings generator
│   ├── retriever.py             # BM25 + Cosine semantic search + re-ranking
│   └── chat.py                  # Conversational RAG CLI
├── main.py                      # Orchestra script
├── requirements.txt             # Python dependencies
└── .env.example                 # Environment template
```

## Setup & Running

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment**:
   Copy `.env.example` to `.env` and fill in your Gemini API key:
   ```bash
   cp .env.example .env
   # Open .env and add your GOOGLE_API_KEY
   ```

3. **Run the pipeline**:
   To run the complete pipeline (extract, embed, and start chatting):
   ```bash
   python main.py all
   ```

   Or run individual commands:
   - `python main.py extract`: Extract text, diagrams, and annotate layouts.
   - `python main.py embed`: Generate local page-wise embeddings.
   - `python main.py chat`: Start the interactive CLI chat session.
