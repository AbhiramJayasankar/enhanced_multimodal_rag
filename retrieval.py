from transformers import AutoModel
import torch
import faiss
import pickle
import numpy as np
import json

def load_model_and_indexes():
    # Load model
    model = AutoModel.from_pretrained("jinaai/jina-embeddings-v4", trust_remote_code=True, dtype=torch.float16)
    model.to("cuda")

    # Load indices and metadata
    query_index = faiss.read_index("storage/query_embeddings.index")
    image_index = faiss.read_index("storage/image_embeddings.index")

    with open("storage/query_metadata.pkl", "rb") as f:
        query_metadata = pickle.load(f)

    with open("storage/image_metadata.pkl", "rb") as f:
        image_metadata = pickle.load(f)

    return model, query_index, image_index, query_metadata, image_metadata

def search(query_text, model, query_index, image_index, query_metadata, image_metadata):
    # Handle both single query and list of queries
    queries = query_text if isinstance(query_text, list) else [query_text]

    # Encode queries
    query_embedding = model.encode_text(
        texts=queries,
        task="retrieval",
    )

    query_embedding_np = np.array([emb.cpu().numpy() for emb in query_embedding])

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
    model, query_index, image_index, query_metadata, image_metadata = load_model_and_indexes()

    # Sample search
    results = search("What is ColPali's average nDCG@5 score across all ViDoRe tasks?",model, query_index, image_index, query_metadata, image_metadata)


    with open("storage/retrieved.json", "w") as f:
        f.write(results)
