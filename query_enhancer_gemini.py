"""Query enhancement powered by Google GenAI SDK and gemini-3.1-flash-lite."""
from __future__ import annotations

import json
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

def enhance_query(query: str) -> dict:
    """
    Takes a query string and returns a JSON dict with enhanced queries.

    Args:
        query: The input query string

    Returns:
        dict: JSON object with enhanced queries
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not found in environment")

    client = genai.Client(api_key=api_key)

    system_prompt = """You are a specialized assistant whose job is to take a raw user query and transform it into multiple enhanced queries optimized for retrieval.

Split the query into sub-queries if it contains multiple intents, conditions, or entities.
Expand and enhance the query by adding synonyms, related terms, and contextually relevant variations that a user might mean.
Preserve factual accuracy—do not hallucinate or add irrelevant entities.
Balance granularity—produce queries that range from highly precise (with key terms preserved) to broader, recall-oriented versions.
Maintain neutrality—do not answer the query, only rewrite/enhance it.
Return a JSON object with a queries array that lists the rewritten sub-queries."""

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=query,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
        )
    )
    
    text = (response.text or "").strip()

    if not text:
        raise RuntimeError("Model response was empty")

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse model response as JSON: {e}")

if __name__ == "__main__":
    query = "What is ColPali's average nDCG@5 score across all ViDoRe tasks?"
    result = enhance_query(query)
    print(json.dumps(result, indent=2))
