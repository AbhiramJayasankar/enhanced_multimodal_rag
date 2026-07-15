import streamlit as st
import sys
import os
import json
from PIL import Image
from dotenv import load_dotenv
from google import genai

# Ensure standard output and error use UTF-8 to prevent encoding crashes
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Gemini Multimodal RAG Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
with st.sidebar:
    st.title("⚙️ RAG Settings")
    st.markdown("""
    This frontend lets you interact with your visual and textual document repository powered by Google's native **Gemini Embeddings 2** (`gemini-embedding-2`) and **Gemini 3.1 Flash Lite**!
    """)
    st.divider()
    st.markdown("### 🗄️ Database Status")
    
    # Check if indices exist
    if os.path.exists("storage/query_embeddings_gemini.index") and os.path.exists("storage/image_embeddings_gemini.index"):
        st.success("✅ Gemini FAISS Indices Loaded")
    else:
        st.error("❌ Gemini FAISS Indices Not Found. Run `build_embeddings_gemini.py` first.")

# Cache the heavy resource loader (FAISS indices + Client) so it loads instantly
@st.cache_resource
def load_rag_backend():
    from retrieval_gemini import load_model_and_indexes
    from query_enhancer_gemini import enhance_query
    
    # Load backend
    client, query_index, image_index, query_metadata, image_metadata = load_model_and_indexes()
    return client, query_index, image_index, query_metadata, image_metadata, enhance_query

# Load resources
try:
    client, query_index, image_index, query_metadata, image_metadata, enhance_query = load_rag_backend()
except Exception as e:
    st.error(f"Failed to load RAG backend resources: {e}")
    st.stop()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Main UI Header
st.title("🤖 Gemini Multimodal RAG Document Chat")
st.markdown("Ask questions about your loaded documents. The system will automatically expand your query, search visual and text databases, and reason across both text and image pages to formulate the perfect answer.")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # If there is retrieval metadata, show it below the message
        if message["role"] == "assistant" and "retrieval_info" in message:
            with st.expander("🔍 Show Retrieved Sources for this query"):
                ret_info = message["retrieval_info"]
                
                # Show enhanced queries
                st.markdown("**Enhanced query variations used for search:**")
                st.write(ret_info["enhanced_queries"])
                
                # Tabs for text and image sources
                tab_text, tab_img = st.tabs(["📝 Text Passages", "🖼️ Visual Pages"])
                
                with tab_text:
                    for idx, text_item in enumerate(ret_info["texts"], 1):
                        st.info(f"**Source Passage {idx}** (Distance: {text_item['distance']:.4f})\n\n{text_item['text']}")
                        
                with tab_img:
                    cols = st.columns(3)
                    for idx, img_item in enumerate(ret_info["images"]):
                        col_idx = idx % 3
                        with cols[col_idx]:
                            try:
                                img = Image.open(img_item["path"])
                                st.image(img, caption=f"Page Image (Dist: {img_item['distance']:.4f})")
                            except Exception as img_err:
                                st.error(f"Error loading image {os.path.basename(img_item['path'])}: {img_err}")

# Check if a query is a casual greeting
def is_greeting(query: str) -> bool:
    q = query.strip().lower().replace("?", "").replace("!", "").replace(".", "")
    greetings = {
        "hi", "hello", "hey", "greetings", "howdy", "good morning", "good afternoon", "good evening", "yo", "sup", "hi there", "hello there"
    }
    return q in greetings

# React to user input
if user_query := st.chat_input("Enter your question here..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Assistant Response Container
    with st.chat_message("assistant"):
        if is_greeting(user_query):
            with st.status("Thinking...", expanded=True) as status_container:
                # Direct response for casual greetings, skipping heavy RAG lookup
                response = client.models.generate_content(
                    model='gemini-3.1-flash-lite',
                    contents=f"The user said '{user_query}' as a casual greeting to a document retrieval RAG chatbot. Respond with a friendly, welcoming greeting and ask how you can help them with their documents today."
                )
                status_container.update(label="Response generated!", state="complete", expanded=False)
            st.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
        else:
            with st.status("Processing RAG pipeline...", expanded=True) as status_container:
                # 1. Enhance Query
                status_container.write("🔮 Expanding and enhancing query...")
                try:
                    enhanced_res = enhance_query(user_query)
                    enhanced_queries = enhanced_res['queries']
                except Exception as e:
                    st.error(f"Query enhancement failed: {e}")
                    enhanced_queries = [user_query]
                
                # 2. Perform Multimodal Retrieval
                status_container.write("🔍 Searching text and visual vector databases...")
                from retrieval_gemini import search
                search_res_json = search(
                    enhanced_queries, 
                    client, 
                    query_index, 
                    image_index, 
                    query_metadata, 
                    image_metadata
                )
                search_results_list = json.loads(search_res_json)
                
                # 3. Aggregate unique matches
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
                
                # 4. Construct Context Text and load Images
                status_container.write("🖼️ Loading visual and textual matches...")
                context_text = "Retrieved context:\n\n"
                for i, text_result in enumerate(all_texts, 1):
                    context_text += f"Text {i}:\n{text_result['text']}\n\n"
                    
                retrieved_images = []
                for img_result in all_images:
                    try:
                        img = Image.open(img_result['path'])
                        retrieved_images.append(img)
                    except Exception as e:
                        pass

                # 5. Synthesize final answer via Gemini
                status_container.write("🧠 Synthesizing final answer with gemini-3.1-flash-lite...")
                full_prompt = f"{context_text}\nUser query: {user_query}"
                contents = retrieved_images + [full_prompt]
                
                response = client.models.generate_content(
                    model='gemini-3.1-flash-lite',
                    contents=contents
                )
                
                status_container.update(label="RAG Pipeline execution complete!", state="complete", expanded=False)

            # Output assistant response
            st.markdown(response.text)
            
            # Display collapsable expander for sources
            with st.expander("🔍 Show Retrieved Sources for this query"):
                st.markdown("**Enhanced query variations used for search:**")
                st.write(enhanced_queries)
                
                tab_text, tab_img = st.tabs(["📝 Text Passages", "🖼️ Visual Pages"])
                with tab_text:
                    for idx, text_item in enumerate(all_texts, 1):
                        st.info(f"**Source Passage {idx}** (Distance: {text_item['distance']:.4f})\n\n{text_item['text']}")
                with tab_img:
                    cols = st.columns(3)
                    for idx, img_item in enumerate(all_images):
                        col_idx = idx % 3
                        with cols[col_idx]:
                            try:
                                img = Image.open(img_item["path"])
                                st.image(img, caption=f"Page Image (Dist: {img_item['distance']:.4f})")
                            except Exception as img_err:
                                st.error(f"Error loading image: {img_err}")
                                
            # Store in history with retrieval metadata
            st.session_state.messages.append({
                "role": "assistant",
                "content": response.text,
                "retrieval_info": {
                    "enhanced_queries": enhanced_queries,
                    "texts": all_texts,
                    "images": all_images
                }
            })
