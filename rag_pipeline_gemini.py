import os
import json
import sys
from dotenv import load_dotenv
from google import genai
from PIL import Image

# Ensure standard output and error use UTF-8 to prevent encoding crashes on Windows consoles
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
if sys.stderr.encoding != 'utf-8':
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Import our Gemini-specific implementations
from retrieval_gemini import load_model_and_indexes, search
from query_enhancer_gemini import enhance_query

load_dotenv()
print("Loaded environment variables\n")

# Initialize Gemini Client
api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)
print("Configured Gemini GenAI Client\n")

print("Loading search model and indexes...")
search_client, query_index, image_index, query_metadata, image_metadata = load_model_and_indexes()
print("Search model and indexes loaded\n")

query = "What is ColPali's average nDCG@5 score across all ViDoRe tasks?"
print(f"Query: {query}\n")

print("Enhancing query...")
enhanced = enhance_query(query)
enhanced_queries = enhanced['queries']
print(f"Enhanced into {len(enhanced_queries)} queries\n")
print(f"Enhanced queries\n")
print(enhanced_queries)


print("Retrieving relevant context using search...")
search_results_json = search(
    enhanced_queries, 
    search_client, 
    query_index, 
    image_index, 
    query_metadata, 
    image_metadata
)
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
contents = retrieved_images + [full_prompt]

# Generate content using gemini-3.1-flash-lite via the google-genai SDK
response = client.models.generate_content(
    model='gemini-3.1-flash-lite',
    contents=contents
)

print("\n\n\n--- Response ---\n")
print(response.text)
