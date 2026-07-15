import os
import json
import glob
import pickle
import threading
import sys
import time
import numpy as np
import faiss
import PIL.Image
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError

# Ensure standard output uses UTF-8 to prevent encoding crashes
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

load_dotenv()

# Setup FastAPI App
app = FastAPI(title="Gemini Multimodal RAG Backend")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
MARKDOWN_DIR = "markdown"
IMAGE_DIR = "images"
STORAGE_DIR = "storage"

for d in [MARKDOWN_DIR, IMAGE_DIR, STORAGE_DIR]:
    os.makedirs(d, exist_ok=True)

# Global task status
process_status = {
    "status": "idle",  # "idle", "running", "completed", "error"
    "message": "",
    "step": "",        # "text", "image"
    "processed": 0,
    "total": 0,
    "error_msg": ""
}

# Lock for indices
rag_resources = {
    "client": None,
    "query_index": None,
    "image_index": None,
    "query_metadata": None,
    "image_metadata": None
}
resources_lock = threading.Lock()

def get_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is not set")
    return genai.Client(api_key=api_key)

def load_resources_safe():
    with resources_lock:
        try:
            client = get_gemini_client()
            rag_resources["client"] = client

            q_idx_path = os.path.join(STORAGE_DIR, "query_embeddings_gemini.index")
            i_idx_path = os.path.join(STORAGE_DIR, "image_embeddings_gemini.index")
            q_meta_path = os.path.join(STORAGE_DIR, "query_metadata_gemini.pkl")
            i_meta_path = os.path.join(STORAGE_DIR, "image_metadata_gemini.pkl")

            if os.path.exists(q_idx_path):
                rag_resources["query_index"] = faiss.read_index(q_idx_path)
            else:
                rag_resources["query_index"] = None

            if os.path.exists(i_idx_path):
                rag_resources["image_index"] = faiss.read_index(i_idx_path)
            else:
                rag_resources["image_index"] = None

            if os.path.exists(q_meta_path):
                with open(q_meta_path, "rb") as f:
                    rag_resources["query_metadata"] = pickle.load(f)
            else:
                rag_resources["query_metadata"] = None

            if os.path.exists(i_meta_path):
                with open(i_meta_path, "rb") as f:
                    rag_resources["image_metadata"] = pickle.load(f)
            else:
                rag_resources["image_metadata"] = None

        except Exception as e:
            print(f"Error loading resources: {e}")

# Initial load
try:
    load_resources_safe()
except Exception:
    pass

# Helper to chunk a markdown file by headings
def chunk_markdown_file(file_path):
    import re
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    chunks = []
    lines = content.split('\n')
    current_chunk = {'heading': '', 'content': ''}

    for line in lines:
        if re.match(r'^#+\s+', line):
            if current_chunk['heading'] or current_chunk['content'].strip():
                chunks.append({
                    'heading': current_chunk['heading'],
                    'content': current_chunk['content'].strip()
                })
            current_chunk = {
                'heading': line.strip(),
                'content': ''
            }
        else:
            current_chunk['content'] += line + '\n'

    if current_chunk['heading'] or current_chunk['content'].strip():
        chunks.append({
            'heading': current_chunk['heading'],
            'content': current_chunk['content'].strip()
        })

    return chunks

# Background thread task for building embeddings
def build_embeddings_task():
    global process_status
    try:
        client_instance = get_gemini_client()
        MODEL_ID = "gemini-embedding-2"

        q_idx_path = os.path.join(STORAGE_DIR, "query_embeddings_gemini.index")
        i_idx_path = os.path.join(STORAGE_DIR, "image_embeddings_gemini.index")
        q_meta_path = os.path.join(STORAGE_DIR, "query_metadata_gemini.pkl")
        i_meta_path = os.path.join(STORAGE_DIR, "image_metadata_gemini.pkl")

        # Load existing index & metadata for incremental lookup
        existing_chunks_map = {}
        if os.path.exists(q_meta_path) and os.path.exists(q_idx_path):
            try:
                print("Loading existing text index for incremental update...")
                with open(q_meta_path, "rb") as f:
                    old_meta = pickle.load(f)
                if "embeddings" in old_meta and "chunks" in old_meta:
                    for idx, old_chunk in enumerate(old_meta["chunks"]):
                        key = (old_chunk.get('heading', ''), old_chunk.get('content', ''))
                        existing_chunks_map[key] = old_meta["embeddings"][idx]
                elif "chunks" in old_meta:
                    try:
                        old_index = faiss.read_index(q_idx_path)
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
        if os.path.exists(i_meta_path) and os.path.exists(i_idx_path):
            try:
                print("Loading existing image index for incremental update...")
                with open(i_meta_path, "rb") as f:
                    old_meta = pickle.load(f)
                if "embeddings" in old_meta and "paths" in old_meta:
                    for idx, path in enumerate(old_meta["paths"]):
                        existing_images_map[os.path.normpath(path)] = old_meta["embeddings"][idx]
                elif "paths" in old_meta:
                    try:
                        old_index = faiss.read_index(i_idx_path)
                        for idx, path in enumerate(old_meta["paths"]):
                            if idx < old_index.ntotal:
                                vec = old_index.reconstruct(idx)
                                existing_images_map[os.path.normpath(path)] = vec
                    except Exception as re_err:
                        print(f"Image index vector reconstruction failed: {re_err}")
                print(f"Loaded {len(existing_images_map)} existing image embeddings.")
            except Exception as e:
                print(f"Could not load existing image index: {e}")

        # 1. Process Text Chunks
        chunks_file_path = os.path.join(STORAGE_DIR, "colpali_chunks.json")
        if not os.path.exists(chunks_file_path):
            raise FileNotFoundError("colpali_chunks.json not found in storage. Please run chunking first.")

        with open(chunks_file_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        texts = [f"{chunk['heading']}\n{chunk['content']}" for chunk in chunks]

        process_status["status"] = "running"
        process_status["message"] = "Building text embeddings..."
        process_status["step"] = "text"
        process_status["processed"] = 0
        process_status["total"] = len(texts)

        query_index = None
        query_embeddings_all = []

        for idx, chunk in enumerate(chunks):
            heading = chunk.get('heading', 'none')
            content = chunk.get('content', '')
            prepared_text = f"title: {heading} | text: {content}"

            key = (heading, content)
            if key in existing_chunks_map:
                emb_values = existing_chunks_map[key]
            else:
                # Call Gemini embedding api with retry
                retries = 5
                for attempt in range(retries):
                    try:
                        response = client_instance.models.embed_content(
                            model=MODEL_ID,
                            contents=prepared_text
                        )
                        emb_values = response.embeddings[0].values
                        break
                    except (APIError, Exception) as e:
                        if attempt == retries - 1:
                            raise e
                        time.sleep(2 ** attempt)
                time.sleep(0.1)

            query_embeddings_all.append(emb_values)
            process_status["processed"] = idx + 1

        query_embeddings_np = np.array(query_embeddings_all, dtype=np.float32)
        query_index = faiss.IndexFlatL2(query_embeddings_np.shape[1])
        query_index.add(query_embeddings_np)

        # 2. Process Images
        image_paths = glob.glob(os.path.join(IMAGE_DIR, "*.png")) + glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
        
        process_status["message"] = "Building image embeddings..."
        process_status["step"] = "image"
        process_status["processed"] = 0
        process_status["total"] = len(image_paths)

        image_index = None
        image_embeddings_all = []

        for idx, img_path in enumerate(image_paths):
            norm_path = os.path.normpath(img_path)
            if norm_path in existing_images_map:
                emb_values = existing_images_map[norm_path]
            else:
                retries = 5
                for attempt in range(retries):
                    try:
                        img = PIL.Image.open(img_path)
                        response = client_instance.models.embed_content(
                            model=MODEL_ID,
                            contents=img
                        )
                        emb_values = response.embeddings[0].values
                        break
                    except (APIError, Exception) as e:
                        if attempt == retries - 1:
                            raise e
                        time.sleep(2 ** attempt)
                time.sleep(0.1)

            image_embeddings_all.append(emb_values)
            process_status["processed"] = idx + 1

        if len(image_embeddings_all) > 0:
            image_embeddings_np = np.array(image_embeddings_all, dtype=np.float32)
            image_index = faiss.IndexFlatL2(image_embeddings_np.shape[1])
            image_index.add(image_embeddings_np)
        else:
            image_index = None

        # 3. Save FAISS index and metadata
        faiss.write_index(query_index, q_idx_path)
        with open(q_meta_path, "wb") as f:
            pickle.dump({
                "texts": texts, 
                "chunks": chunks, 
                "embeddings": [list(v) for v in query_embeddings_all]
            }, f)

        if image_index is not None:
            faiss.write_index(image_index, i_idx_path)
            with open(i_meta_path, "wb") as f:
                pickle.dump({
                    "paths": image_paths, 
                    "embeddings": [list(v) for v in image_embeddings_all]
                }, f)

        # Reload resources for RAG chat
        load_resources_safe()

        process_status["status"] = "completed"
        process_status["message"] = "Embeddings built successfully!"

    except Exception as e:
        process_status["status"] = "error"
        process_status["message"] = "Embedding generation failed."
        process_status["error_msg"] = str(e)
        print(f"Embedding building task failed: {e}")

# API Endpoints
@app.get("/api/status")
def get_system_status():
    """
    Returns system stats (number of files, index counts, etc.)
    """
    markdown_files = [os.path.basename(f) for f in glob.glob(os.path.join(MARKDOWN_DIR, "*.md"))]
    image_files = [os.path.basename(f) for f in glob.glob(os.path.join(IMAGE_DIR, "*.png")) + glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))]

    chunks_file = os.path.join(STORAGE_DIR, "colpali_chunks.json")
    chunks_count = 0
    if os.path.exists(chunks_file):
        try:
            with open(chunks_file, "r", encoding="utf-8") as f:
                chunks_count = len(json.load(f))
        except Exception:
            pass

    indexed_texts = 0
    indexed_images = 0

    with resources_lock:
        if rag_resources["query_index"] is not None:
            indexed_texts = rag_resources["query_index"].ntotal
        if rag_resources["image_index"] is not None:
            indexed_images = rag_resources["image_index"].ntotal

    return {
        "markdown_files": markdown_files,
        "image_files": image_files,
        "chunks_count": chunks_count,
        "indexed_texts": indexed_texts,
        "indexed_images": indexed_images,
        "api_configured": os.environ.get("GEMINI_API_KEY") is not None or os.environ.get("GOOGLE_API_KEY") is not None
    }

@app.post("/api/upload/markdown")
async def upload_markdown(file: UploadFile = File(...)):
    """
    Saves uploaded markdown file
    """
    if not file.filename.endswith(".md"):
        raise HTTPException(status_code=400, detail="Only markdown (.md) files are supported")
    
    file_path = os.path.join(MARKDOWN_DIR, file.filename)
    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        return {"filename": file.filename, "message": "Uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

@app.post("/api/upload/image")
async def upload_image(file: UploadFile = File(...)):
    """
    Saves uploaded image file
    """
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".png", ".jpg", ".jpeg"]:
        raise HTTPException(status_code=400, detail="Only .png, .jpg, and .jpeg files are supported")
    
    file_path = os.path.join(IMAGE_DIR, file.filename)
    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        return {"filename": file.filename, "message": "Uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

@app.post("/api/process/chunk")
def run_chunking():
    """
    Chunks all markdown files in directory and combines them in storage/colpali_chunks.json
    """
    markdown_files = glob.glob(os.path.join(MARKDOWN_DIR, "*.md"))
    if not markdown_files:
        raise HTTPException(status_code=400, detail="No markdown files found to chunk")

    all_chunks = []
    try:
        for f in markdown_files:
            file_chunks = chunk_markdown_file(f)
            all_chunks.extend(file_chunks)

        out_path = os.path.join(STORAGE_DIR, "colpali_chunks.json")
        with open(out_path, "w", encoding="utf-8") as out:
            json.dump(all_chunks, out, indent=2, ensure_ascii=False)

        return {"status": "success", "chunks_count": len(all_chunks), "message": f"Successfully chunked {len(all_chunks)} sections."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chunking failed: {e}")

@app.post("/api/process/embed")
def start_embedding(background_tasks: BackgroundTasks):
    """
    Triggers background task to build vector indices
    """
    global process_status
    if process_status["status"] == "running":
        return {"status": "already_running", "message": "Embeddings are currently being built."}

    # Reset status
    process_status = {
        "status": "running",
        "message": "Initializing embedding builder task...",
        "step": "init",
        "processed": 0,
        "total": 0,
        "error_msg": ""
    }

    # Run in background
    background_tasks.add_task(build_embeddings_task)
    return {"status": "started", "message": "Embedding process started in background."}

@app.get("/api/process/status")
def get_process_status():
    """
    Returns active task running progress
    """
    return process_status

class ChatQuery(BaseModel):
    query: str

@app.post("/api/chat")
def run_chat_query(chat_data: ChatQuery):
    """
    Executes RAG Chat synthesis using query expansion, vector search, and Gemini
    """
    query = chat_data.query
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    from query_enhancer_gemini import enhance_query
    from retrieval_gemini import search

    # Extract active resources
    with resources_lock:
        client = rag_resources["client"]
        query_index = rag_resources["query_index"]
        image_index = rag_resources["image_index"]
        query_metadata = rag_resources["query_metadata"]
        image_metadata = rag_resources["image_metadata"]

    if not client or query_index is None or query_metadata is None:
        raise HTTPException(status_code=400, detail="FAISS Vector Indices are not loaded. Please build the embeddings first.")

    try:
        # Check if it is a casual greeting
        q_cleaned = query.strip().lower().replace("?", "").replace("!", "").replace(".", "")
        greetings = {"hi", "hello", "hey", "greetings", "howdy", "good morning", "good afternoon", "good evening", "yo", "sup", "hi there", "hello there"}
        
        if q_cleaned in greetings:
            response = client.models.generate_content(
                model='gemini-3.1-flash-lite',
                contents=f"The user said '{query}' as a casual greeting to a document retrieval RAG chatbot. Respond with a friendly, welcoming greeting and ask how you can help them with their documents today."
            )
            return {
                "answer": response.text,
                "enhanced_queries": [query],
                "texts": [],
                "images": []
            }

        # 1. Enhance Query
        try:
            enhanced_res = enhance_query(query)
            enhanced_queries = enhanced_res['queries']
        except Exception:
            enhanced_queries = [query]

        # 2. Vector Search
        search_res_json = search(
            enhanced_queries,
            client,
            query_index,
            image_index,
            query_metadata,
            image_metadata
        )
        search_results_list = json.loads(search_res_json)

        # 3. Aggregate unique results
        all_texts = []
        all_images = []
        seen_texts = set()
        seen_images = set()

        for result in search_results_list:
            for text_result in result['texts']:
                text_content = text_result['text']
                if text_content not in seen_texts:
                    seen_texts.add(text_content)
                    all_texts.append(text_result)

            for img_result in result['images']:
                img_path = img_result['path']
                if img_path not in seen_images:
                    seen_images.add(img_path)
                    all_images.append(img_result)

        # 4. Construct prompt and load images
        context_text = "Retrieved context:\n\n"
        for i, text_result in enumerate(all_texts, 1):
            context_text += f"Text {i}:\n{text_result['text']}\n\n"

        retrieved_images = []
        for img_result in all_images:
            try:
                # Keep path relative to workspace or return base path
                img = PIL.Image.open(img_result['path'])
                retrieved_images.append(img)
            except Exception:
                pass

        full_prompt = f"{context_text}\nUser query: {query}"
        contents = retrieved_images + [full_prompt]

        # 5. Synthesis
        response = client.models.generate_content(
            model='gemini-3.1-flash-lite',
            contents=contents
        )

        # Clean images paths for frontend (just filename to be loaded from image static dir)
        cleaned_images = []
        for img in all_images:
            cleaned_images.append({
                "filename": os.path.basename(img["path"]),
                "distance": img["distance"]
            })

        return {
            "answer": response.text,
            "enhanced_queries": enhanced_queries,
            "texts": all_texts,
            "images": cleaned_images
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query execution failed: {e}")

# Static directory route to serve loaded images to frontend
from fastapi.staticfiles import StaticFiles
app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
