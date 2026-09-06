import asyncio
import os
import threading
import requests

from typing import TypedDict, Annotated

from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool, BaseTool

from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun

from langchain_mcp_adapters.client import MultiServerMCPClient

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import aiosqlite


# ============================================================
# 1. Load Environment Variables
# ============================================================

load_dotenv()


# ============================================================
# 2. Dedicated Async Event Loop
# ============================================================

_ASYNC_LOOP = asyncio.new_event_loop()

_ASYNC_THREAD = threading.Thread(
    target=_ASYNC_LOOP.run_forever,
    daemon=True
)

_ASYNC_THREAD.start()


def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(
        coro,
        _ASYNC_LOOP
    )


def run_async(coro):
    return _submit_async(coro).result()


def submit_async_task(coro):
    """
    Schedule a coroutine on the backend event loop.
    """
    return _submit_async(coro)


# ============================================================
# 3. ChatGroq LLM
# ============================================================

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# ============================================================
# 4. Web Search Tool
# ============================================================

search_tool = DuckDuckGoSearchRun(
    region="us-en"
)


# ============================================================
# 5. Stock Price Tool
# ============================================================

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest stock price for a given stock symbol.

    Examples:
    AAPL
    TSLA
    MSFT
    """

    api_key = os.getenv(
        "ALPHAVANTAGE_API_KEY"
    )

    if not api_key:

        return {
            "error": (
                "ALPHAVANTAGE_API_KEY "
                "is not configured in .env"
            )
        }

    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE"
        f"&symbol={symbol}"
        f"&apikey={api_key}"
    )

    try:

        response = requests.get(
            url,
            timeout=10
        )

        response.raise_for_status()

        return response.json()

    except Exception as e:

        return {
            "error": str(e)
        }


# ============================================================
# 6. Your Own MCP Server
# ============================================================

client = MultiServerMCPClient(
    {
        "my_mcp": {
            "transport": "streamable_http",
            "url": "http://127.0.0.1:8000/mcp"
        }
    }
)


# ============================================================
# 7. Load MCP Tools
# ============================================================

def load_mcp_tools() -> list[BaseTool]:

    try:

        tools = run_async(
            client.get_tools()
        )

        print(
            f"Loaded {len(tools)} MCP tools."
        )

        return tools

    except Exception as e:

        print(
            f"Could not load MCP tools: {e}"
        )

        return []


mcp_tools = load_mcp_tools()


# ============================================================
# 8. Combine All Tools
# ============================================================

tools = [
    search_tool,
    get_stock_price,
    *mcp_tools
]


print(
    "Available tools:",
    [tool.name for tool in tools]
)


# ============================================================
# 9. Bind Tools to ChatGroq
# ============================================================

if tools:

    llm_with_tools = model.bind_tools(
        tools
    )

else:

    llm_with_tools = model


# ============================================================
# 10. State
# ============================================================

class ChatState(TypedDict):

    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]


# ============================================================
# 11. Chat Node
# ============================================================

async def chat_node(
    state: ChatState
):

    messages = state["messages"]

    response = await llm_with_tools.ainvoke(
        messages
    )

    return {
        "messages": [response]
    }


# ============================================================
# 12. Tool Node
# ============================================================

tool_node = (
    ToolNode(tools)
    if tools
    else None
)


# ============================================================
# 13. SQLite Checkpointer
# ============================================================

async def _init_checkpointer():

    conn = await aiosqlite.connect(
        database="chatbot.db"
    )

    return AsyncSqliteSaver(
        conn
    )


checkpointer = run_async(
    _init_checkpointer()
)


# ============================================================
# 14. Create LangGraph
# ============================================================

graph = StateGraph(
    ChatState
)


graph.add_node(
    "chat_node",
    chat_node
)


graph.add_edge(
    START,
    "chat_node"
)


# ============================================================
# 15. Tool Routing
# ============================================================

if tool_node:

    graph.add_node(
        "tools",
        tool_node
    )

    graph.add_conditional_edges(
        "chat_node",
        tools_condition
    )

    graph.add_edge(
        "tools",
        "chat_node"
    )

else:

    graph.add_edge(
        "chat_node",
        END
    )


# ============================================================
# 16. Compile Graph
# ============================================================

chatbot = graph.compile(
    checkpointer=checkpointer
)


# ============================================================
# 17. Retrieve All Threads
# ============================================================

async def _alist_threads():

    all_threads = set()

    async for checkpoint in checkpointer.alist(None):

        thread_id = checkpoint.config[
            "configurable"
        ].get(
            "thread_id"
        )

        if thread_id:

            all_threads.add(
                thread_id
            )

    return list(all_threads)


def retrieve_all_threads():

    return run_async(
        _alist_threads()
    )