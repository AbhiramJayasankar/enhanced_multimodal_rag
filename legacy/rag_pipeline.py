import os
from dotenv import load_dotenv
import google.generativeai as genai
import json
from retrieval import load_model_and_indexes, search
from query_enhancer import enhance_query
from PIL import Image

load_dotenv()
print("Loaded environment variables\n")

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
print("Configured Gemini API\n")

print("Loading search model and indexes...")
search_model, query_index, image_index, query_metadata, image_metadata = load_model_and_indexes()
print("Search model and indexes loaded\n")

gemini_model = genai.GenerativeModel('gemini-3.1-flash-lite')
print("Initialized Gemini model\n")

query = "What is ColPali's average nDCG@5 score across all ViDoRe tasks?"
print(f"Query: {query}\n")

print("Enhancing query...")
enhanced = enhance_query(query)
enhanced_queries = enhanced['queries']
print(f"Enhanced into {len(enhanced_queries)} queries\n")
print(f"Enhanced queries\n")
print(enhanced_queries)


print("Retrieving relevant context using search...")
search_results_json = search(enhanced_queries, search_model, query_index, image_index, query_metadata, image_metadata)
search_results_list = json.loads(search_results_json)
print(f"Retrieved results for {len(search_results_list)} queries\n")

# Aggregate results from all enhanced queries
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

print(f"Aggregated {len(all_texts)} unique texts and {len(all_images)} unique images\n")

print("Retrieved texts:")
for i, text_result in enumerate(all_texts, 1):
    print(f"Text {i}: {text_result['text']}")
print()

print("Retrieved image files:")
for i, img_result in enumerate(all_images, 1):
    print(f"Image {i}: {os.path.basename(img_result['path'])}")
print()

context_text = "Retrieved context:\n\n"
for i, text_result in enumerate(all_texts, 1):
    context_text += f"Text {i}:\n{text_result['text']}\n\n"

print("Loading retrieved images...")
retrieved_images = []
for img_result in all_images:
    try:
        img = Image.open(img_result['path'])
        retrieved_images.append(img)
    except Exception as e:
        print(f"Could not load image {img_result['path']}: {e}")
print(f"Loaded {len(retrieved_images)} images\n")

full_prompt = f"{context_text}\nUser query: {query}"
print("Built prompt with context\n")

print("Sending to Gemini...")
content = retrieved_images + [full_prompt]
response = gemini_model.generate_content(content)

print("\n\n\n--- Response ---\n")
print(response.text)
