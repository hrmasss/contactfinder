from google import genai
from google.genai import types


def gemini(prompt: str) -> str:
    """Gemini 2.0 Flash with Google Search."""
    client = genai.Client()
    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    config = types.GenerateContentConfig(tools=[grounding_tool])
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=config,
    )
    return response.text
