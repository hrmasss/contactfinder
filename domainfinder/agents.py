from google import genai
from google.genai import types
from langchain_core.tools import Tool
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langchain_community.utilities import GoogleSerperAPIWrapper


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


def gpt(prompt: str, model: str = "gpt-4o-mini") -> str:
    """GPT agent with search capabilities using langchain"""

    llm = init_chat_model(model, model_provider="openai", temperature=0)
    search = GoogleSerperAPIWrapper()

    tools = [
        Tool(
            name="search",
            func=search.run,
            description="useful for when you need to search for information",
        )
    ]

    agent = create_react_agent(llm, tools)

    events = agent.stream(
        {"messages": [("user", prompt)]},
        stream_mode="values",
    )

    # Get the last message from the agent
    for event in events:
        pass

    return event["messages"][-1].content
