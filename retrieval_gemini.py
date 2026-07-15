from google import genai
from google.genai.errors import APIError
import faiss
import pickle
import numpy as np
import json
import os
import time
from dotenv import load_dotenv

load_dotenv()

def load_model_and_indexes():
    """
    Initializes the Gemini client and loads the Gemini FAISS indexes and metadata.
    """
    # Initialize Client
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set")
    
    client = genai.Client(api_key=api_key)

    # Load indices and metadata
    query_index = faiss.read_index("storage/query_embeddings_gemini.index")
    image_index = faiss.read_index("storage/image_embeddings_gemini.index")

    with open("storage/query_metadata_gemini.pkl", "rb") as f:
        query_metadata = pickle.load(f)

    with open("storage/image_metadata_gemini.pkl", "rb") as f:
        image_metadata = pickle.load(f)

    return client, query_index, image_index, query_metadata, image_metadata

def search(query_text, client, query_index, image_index, query_metadata, image_metadata):
    """
    Encodes the given text queries using gemini-embedding-2 and retrieves
    the most relevant text and image indices using FAISS.
    """
    # Handle both single query and list of queries
    queries = query_text if isinstance(query_text, list) else [query_text]

    # Encode queries using gemini-embedding-2
    query_embeddings_list = []
    for q in queries:
        # Prefix the query according to Google's best practices for asymmetric retrieval:
        # f"task: question answering | query: {query}"
        prepared_query = f"task: question answering | query: {q}"
        
        # Exponential backoff retry logic for robust API embedding requests
        retries = 5
        backoff_factor = 2
        emb_values = None
        for attempt in range(retries):
            try:
                response = client.models.embed_content(
                    model="gemini-embedding-2",
                    contents=prepared_query,
                )
                emb_values = response.embeddings[0].values
                break
            except APIError as e:
                if attempt == retries - 1:
                    raise e
                sleep_time = backoff_factor ** attempt
                print(f"API Error during search query embedding: {e}. Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            except Exception as e:
                if attempt == retries - 1:
                    raise e
                sleep_time = backoff_factor ** attempt
                print(f"Error during search query embedding: {e}. Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
                
        query_embeddings_list.append(emb_values)
        time.sleep(0.1)  # Safe rate-limit buffer

    query_embedding_np = np.array(query_embeddings_list, dtype=np.float32)

    # Search text and image indexes for all queries
    k = 5
    text_distances, text_indices = query_index.search(query_embedding_np, k)
    image_distances, image_indices = image_index.search(query_embedding_np, k)

    # Build result JSON
    results = []
    for query_idx, query in enumerate(queries):
        result = {
            "query": query,
            "texts": [
                {
                    "text": query_metadata['texts'][idx],
                    "distance": float(text_distances[query_idx][i])
                }
                for i, idx in enumerate(text_indices[query_idx])
            ],
            "images": [
                {
                    "path": image_metadata['paths'][idx],
                    "distance": float(image_distances[query_idx][i])
                }
                for i, idx in enumerate(image_indices[query_idx])
            ]
        }
        results.append(result)

    # Return single result if single query, otherwise return list
    return json.dumps(results[0] if len(results) == 1 else results, indent=2)

if __name__ == "__main__":
    # Load model and indexes
    client, query_index, image_index, query_metadata, image_metadata = load_model_and_indexes()

    # Sample search
    results = search(
        "What is ColPali's average nDCG@5 score across all ViDoRe tasks?",
        client, 
        query_index, 
        image_index, 
        query_metadata, 
        image_metadata
    )

    print("Sample Search Results:")
    print(results)
