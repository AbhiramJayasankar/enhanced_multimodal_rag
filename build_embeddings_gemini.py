from google import genai
from google.genai import types
from google.genai.errors import APIError
import glob
import os
import json
import faiss
import pickle
import numpy as np
import time
import PIL.Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Gemini Client
# The client automatically picks up GEMINI_API_KEY (or GOOGLE_API_KEY) from the environment
api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set in .env")

client = genai.Client(api_key=api_key)

# Configuration
MODEL_ID = "gemini-embedding-2"
BATCH_SIZE = 1
IMAGE_BATCH_SIZE = 1

# Output paths (using gemini suffix to prevent overwriting original local Jina indices)
QUERY_INDEX_PATH = "storage/query_embeddings_gemini.index"
QUERY_METADATA_PATH = "storage/query_metadata_gemini.pkl"
IMAGE_INDEX_PATH = "storage/image_embeddings_gemini.index"
IMAGE_METADATA_PATH = "storage/image_metadata_gemini.pkl"

# Ensure storage directory exists
os.makedirs("storage", exist_ok=True)

def get_text_embedding_with_retry(text, retries=5, backoff_factor=2):
    """
    Get text embedding from Gemini API with exponential backoff retry.
    """
    for attempt in range(retries):
        try:
            response = client.models.embed_content(
                model=MODEL_ID,
                contents=text,
            )
            return response.embeddings[0].values
        except APIError as e:
            if attempt == retries - 1:
                raise e
            sleep_time = backoff_factor ** attempt
            print(f"API Error: {e}. Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)
        except Exception as e:
            if attempt == retries - 1:
                raise e
            sleep_time = backoff_factor ** attempt
            print(f"Error: {e}. Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)

def get_image_embedding_with_retry(image_path, retries=5, backoff_factor=2):
    """
    Get image embedding from Gemini API with exponential backoff retry.
    """
    for attempt in range(retries):
        try:
            img = PIL.Image.open(image_path)
            response = client.models.embed_content(
                model=MODEL_ID,
                contents=img,
            )
            return response.embeddings[0].values
        except APIError as e:
            if attempt == retries - 1:
                raise e
            sleep_time = backoff_factor ** attempt
            print(f"API Error for {image_path}: {e}. Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)
        except Exception as e:
            if attempt == retries - 1:
                raise e
            sleep_time = backoff_factor ** attempt
            print(f"Error for {image_path}: {e}. Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)

# Load text chunks
with open("storage/colpali_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Original formatted texts to match structure of build_embeddings.py metadata
texts = [f"{chunk['heading']}\n{chunk['content']}" for chunk in chunks]

# --- Incremental Indexing Load ---
existing_chunks_map = {}
if os.path.exists(QUERY_METADATA_PATH) and os.path.exists(QUERY_INDEX_PATH):
    try:
        print("Loading existing text index for incremental update...")
        with open(QUERY_METADATA_PATH, "rb") as f:
            old_meta = pickle.load(f)
        if "embeddings" in old_meta and "chunks" in old_meta:
            for idx, old_chunk in enumerate(old_meta["chunks"]):
                key = (old_chunk.get('heading', ''), old_chunk.get('content', ''))
                existing_chunks_map[key] = old_meta["embeddings"][idx]
        elif "chunks" in old_meta:
            try:
                old_index = faiss.read_index(QUERY_INDEX_PATH)
                for idx, old_chunk in enumerate(old_meta["chunks"]):
                    if idx < old_index.ntotal:
                        vec = old_index.reconstruct(idx)
                        key = (old_chunk.get('heading', ''), old_chunk.get('content', ''))
                        existing_chunks_map[key] = vec
            except Exception as re_err:
                print(f"Text index vector reconstruction failed: {re_err}")
        print(f"Loaded {len(existing_chunks_map)} existing text embeddings.")
    except Exception as e:
        print(f"Could not load existing text index: {e}")

existing_images_map = {}
if os.path.exists(IMAGE_METADATA_PATH) and os.path.exists(IMAGE_INDEX_PATH):
    try:
        print("Loading existing image index for incremental update...")
        with open(IMAGE_METADATA_PATH, "rb") as f:
            old_meta = pickle.load(f)
        if "embeddings" in old_meta and "paths" in old_meta:
            for idx, path in enumerate(old_meta["paths"]):
                existing_images_map[os.path.normpath(path)] = old_meta["embeddings"][idx]
        elif "paths" in old_meta:
            try:
                old_index = faiss.read_index(IMAGE_INDEX_PATH)
                for idx, path in enumerate(old_meta["paths"]):
                    if idx < old_index.ntotal:
                        vec = old_index.reconstruct(idx)
                        existing_images_map[os.path.normpath(path)] = vec
            except Exception as re_err:
                print(f"Image index vector reconstruction failed: {re_err}")
        print(f"Loaded {len(existing_images_map)} existing image embeddings.")
    except Exception as e:
        print(f"Could not load existing image index: {e}")

query_index = None
query_embeddings_all = []

print(f"Processing {len(texts)} texts in batches of {BATCH_SIZE}")
for i in range(0, len(texts), BATCH_SIZE):
    batch_chunks = chunks[i:i+BATCH_SIZE]
    print(f"Processing text batch {i//BATCH_SIZE + 1}/{(len(texts)-1)//BATCH_SIZE + 1}")

    batch_embeddings_list = []
    for chunk in batch_chunks:
        # Use Google's recommended document prefixing structure for asymmetric retrieval tasks:
        # "title: {title} | text: {content}"
        heading = chunk.get('heading', 'none')
        content = chunk.get('content', '')
        prepared_text = f"title: {heading} | text: {content}"

        key = (heading, content)
        if key in existing_chunks_map:
            emb_values = existing_chunks_map[key]
        else:
            emb_values = get_text_embedding_with_retry(prepared_text)
            time.sleep(0.1)  # Brief rate-limiting safeguard
            
        batch_embeddings_list.append(emb_values)
        query_embeddings_all.append(emb_values)

    batch_embeddings_np = np.array(batch_embeddings_list, dtype=np.float32)

    # Initialize index with first batch
    if query_index is None:
        query_dimension = batch_embeddings_np.shape[1]
        query_index = faiss.IndexFlatL2(query_dimension)

    query_index.add(batch_embeddings_np)

# Process image embeddings in batches
image_paths = glob.glob("images/*.png") + glob.glob("images/*.jpg")
print(f"Found {len(image_paths)} images")

image_index = None
image_embeddings_all = []

print(f"Processing {len(image_paths)} images in batches of {IMAGE_BATCH_SIZE}")
for i in range(0, len(image_paths), IMAGE_BATCH_SIZE):
    batch_images = image_paths[i:i+IMAGE_BATCH_SIZE]
    print(f"Processing image batch {i//IMAGE_BATCH_SIZE + 1}/{(len(image_paths)-1)//IMAGE_BATCH_SIZE + 1}")

    batch_embeddings_list = []
    for img_path in batch_images:
        norm_path = os.path.normpath(img_path)
        if norm_path in existing_images_map:
            emb_values = existing_images_map[norm_path]
        else:
            emb_values = get_image_embedding_with_retry(img_path)
            time.sleep(0.1)  # Brief rate-limiting safeguard
            
        batch_embeddings_list.append(emb_values)
        image_embeddings_all.append(emb_values)

    batch_embeddings_np = np.array(batch_embeddings_list, dtype=np.float32)

    # Initialize index with first batch
    if image_index is None:
        image_dimension = batch_embeddings_np.shape[1]
        image_index = faiss.IndexFlatL2(image_dimension)

    image_index.add(batch_embeddings_np)

# Save query index and metadata
faiss.write_index(query_index, QUERY_INDEX_PATH)
with open(QUERY_METADATA_PATH, "wb") as f:
    pickle.dump({
        "texts": texts, 
        "chunks": chunks, 
        "embeddings": [list(v) for v in query_embeddings_all]
    }, f)

# Save image index and metadata
faiss.write_index(image_index, IMAGE_INDEX_PATH)
with open(IMAGE_METADATA_PATH, "wb") as f:
    pickle.dump({
        "paths": image_paths, 
        "embeddings": [list(v) for v in image_embeddings_all]
    }, f)

print(f"Saved {len(texts)} query embeddings to {QUERY_INDEX_PATH}")
print(f"Saved {len(image_paths)} image embeddings to {IMAGE_INDEX_PATH}")
