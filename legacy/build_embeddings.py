from transformers import AutoModel
import torch
import glob
import os
import json
import faiss
import pickle
import numpy as np

model = AutoModel.from_pretrained("jinaai/jina-embeddings-v4", trust_remote_code=True, dtype=torch.float16)

model.to("cuda")

with open("storage/colpali_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

texts = [f"{chunk['heading']}\n{chunk['content']}" for chunk in chunks]

# Process text embeddings in batches
BATCH_SIZE = 1
query_index = None

print(f"Processing {len(texts)} texts in batches of {BATCH_SIZE}")
for i in range(0, len(texts), BATCH_SIZE):
    batch_texts = texts[i:i+BATCH_SIZE]
    print(f"Processing text batch {i//BATCH_SIZE + 1}/{(len(texts)-1)//BATCH_SIZE + 1}")

    batch_embeddings = model.encode_text(
        texts=batch_texts,
        task="retrieval",
    )

    batch_embeddings_np = np.array([emb.cpu().numpy() for emb in batch_embeddings])

    # Initialize index with first batch
    if query_index is None:
        query_dimension = batch_embeddings_np.shape[1]
        query_index = faiss.IndexFlatL2(query_dimension)

    query_index.add(batch_embeddings_np)

    # Clear VRAM
    del batch_embeddings
    torch.cuda.empty_cache()

# Process image embeddings in batches
image_paths = glob.glob("images/*.png") + glob.glob("images/*.jpg")
print(f"Found {len(image_paths)} images")

IMAGE_BATCH_SIZE = 1
image_index = None

print(f"Processing {len(image_paths)} images in batches of {IMAGE_BATCH_SIZE}")
for i in range(0, len(image_paths), IMAGE_BATCH_SIZE):
    batch_images = image_paths[i:i+IMAGE_BATCH_SIZE]
    print(f"Processing image batch {i//IMAGE_BATCH_SIZE + 1}/{(len(image_paths)-1)//IMAGE_BATCH_SIZE + 1}")

    batch_embeddings = model.encode_image(
        images=batch_images,
        task="retrieval",
    )

    batch_embeddings_np = np.array([emb.cpu().numpy() for emb in batch_embeddings])

    # Initialize index with first batch
    if image_index is None:
        image_dimension = batch_embeddings_np.shape[1]
        image_index = faiss.IndexFlatL2(image_dimension)

    image_index.add(batch_embeddings_np)

    # Clear VRAM
    del batch_embeddings
    torch.cuda.empty_cache()

# Save query index and metadata
faiss.write_index(query_index, "storage/query_embeddings.index")
with open("storage/query_metadata.pkl", "wb") as f:
    pickle.dump({"texts": texts, "chunks": chunks}, f)

# Save image index and metadata
faiss.write_index(image_index, "storage/image_embeddings.index")
with open("storage/image_metadata.pkl", "wb") as f:
    pickle.dump({"paths": image_paths}, f)

print(f"Saved {len(texts)} query embeddings to query_embeddings.index")
print(f"Saved {len(image_paths)} image embeddings to image_embeddings.index")
