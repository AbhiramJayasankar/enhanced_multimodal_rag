# Multimodal RAG System with Jina Embeddings

A Retrieval-Augmented Generation (RAG) pipeline that combines text and image retrieval using Jina AI's multimodal embeddings (v4) with Google's Gemini for enhanced query processing and answer generation.

## Features

- **Multimodal Embeddings**: Uses Jina Embeddings v4 to encode both text and images into a shared embedding space
- **Query Enhancement**: Leverages Gemini to expand and enhance user queries for better retrieval
- **Hybrid Retrieval**: Searches across both text chunks and images simultaneously
- **RAG Pipeline**: Combines retrieved context (text + images) with Gemini for accurate answer generation
- **Markdown Processing**: Automatically chunks markdown documents by headings for better context

## Project Structure

```
.
├── build_embeddings.py    # Builds and saves FAISS indexes for text and images
├── markdown_chunker.py    # Chunks markdown files by headings
├── query_enhancer.py      # Enhances queries using Gemini
├── retrieval.py           # Handles search/retrieval operations
├── rag_pipeline.py        # Main RAG pipeline orchestration
├── storage/               # Stores embeddings, indexes, and metadata
├── images/                # Source images for embedding
├── markdown/              # Source markdown files
└── reference/             # Reference materials
```

## Setup

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for embeddings)
- Google AI Studio API key (Gemini)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/AbhiramJayasankar/enhanced_multimodal_rag
cd enhanced_multimodal_rag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create a .env file
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

## Usage

### 1. Prepare Your Data

Place your markdown files in the `markdown/` directory and images in the `images/` directory.

### 2. Chunk Markdown Files

```bash
python markdown_chunker.py
```

This will process markdown files and create `storage/colpali_chunks.json` with text chunks organized by headings.

### 3. Build Embeddings

```bash
python build_embeddings.py
```

This will:
- Generate embeddings for all text chunks
- Generate embeddings for all images
- Create FAISS indexes for efficient similarity search
- Save everything to the `storage/` directory

### 4. Run the RAG Pipeline

```bash
python rag_pipeline.py
```

Or use the retrieval module directly:

```python
from retrieval import load_model_and_indexes, search

# Load model and indexes
model, query_index, image_index, query_metadata, image_metadata = load_model_and_indexes()

# Search
results = search(
    "Your query here",
    model,
    query_index,
    image_index,
    query_metadata,
    image_metadata
)
```

### 5. Query Enhancement (Optional)

```python
from query_enhancer import enhance_query

# Enhance a single query into multiple variations
enhanced = enhance_query("What is ColPali's average nDCG@5 score?")
print(enhanced['queries'])
```

## How It Works

### Pipeline Flow

1. **Query Enhancement**: User query is enhanced using Gemini to generate multiple query variations
2. **Embedding**: Enhanced queries are embedded using Jina Embeddings v4
3. **Retrieval**: FAISS searches both text and image indexes for top-k similar items
4. **Aggregation**: Results from all query variations are deduplicated and combined
5. **Generation**: Retrieved context (text + images) is sent to Gemini for final answer generation

### Storage Format

The `storage/` directory contains:
- `colpali_chunks.json` - Chunked text data
- `query_embeddings.index` - FAISS index for text embeddings
- `query_metadata.pkl` - Metadata for text chunks
- `image_embeddings.index` - FAISS index for image embeddings
- `image_metadata.pkl` - Image paths and metadata
- `retrieved.json` - Latest retrieval results

## Configuration

### Batch Sizes

Adjust batch sizes in `build_embeddings.py` based on your GPU memory:

```python
BATCH_SIZE = 1          # Text embedding batch size
IMAGE_BATCH_SIZE = 1    # Image embedding batch size
```

### Retrieval Parameters

Modify the number of results in `retrieval.py`:

```python
k = 5  # Number of top results to retrieve
```

### Model Selection

Change models in respective files:
- **Jina Embeddings**: `jinaai/jina-embeddings-v4` (in `build_embeddings.py` and `retrieval.py`)
- **Gemini Model**: `gemini-3.1-flash-lite` (in `rag_pipeline.py` and `query_enhancer.py`)

## Requirements

See [requirements.txt](requirements.txt) for full dependencies:
- `torch` - PyTorch for deep learning
- `transformers` - Hugging Face transformers for Jina embeddings
- `faiss-cpu/faiss-gpu` - Efficient similarity search
- `google-generativeai` - Gemini API
- `python-dotenv` - Environment variable management
- `Pillow` - Image processing

## Notes

- **GPU Recommended**: Embedding generation is much faster on GPU
- **FAISS Variant**: Use `faiss-gpu` instead of `faiss-cpu` if you have CUDA
- **VRAM Management**: Batch processing and cache clearing prevent OOM errors
- **Incremental Updates**: Re-run `build_embeddings.py` when adding new documents or images

## License

[Add your license here]

## Acknowledgments

- [Jina AI](https://jina.ai/) - Jina Embeddings v4
- [Google](https://ai.google.dev/) - Gemini API
- [FAISS](https://github.com/facebookresearch/faiss) - Similarity search
