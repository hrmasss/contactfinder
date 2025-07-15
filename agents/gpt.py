from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langchain_community.utilities import GoogleSerperAPIWrapper


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
