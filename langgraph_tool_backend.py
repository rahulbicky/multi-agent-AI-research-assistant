# backend.py

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from dotenv import load_dotenv
import sqlite3
import requests
import os

load_dotenv()


# -------------------
# 1. Groq model
# -------------------

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# -------------------
# 2. Tools
# -------------------

search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(
    first_num: float,
    second_num: float,
    operation: str
) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num

        elif operation == "sub":
            result = first_num - second_num

        elif operation == "mul":
            result = first_num * second_num

        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}

            result = first_num / second_num

        else:
            return {
                "error": f"Unsupported operation '{operation}'"
            }

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result
        }

    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage.
    """

    api_key = os.getenv("ALPHAVANTAGE_API_KEY")

    url = (
        f"https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE"
        f"&symbol={symbol}"
        f"&apikey={api_key}"
    )

    r = requests.get(url)

    return r.json()


tools = [
    search_tool,
    get_stock_price,
    calculator
]

# IMPORTANT:
# Use `model`, because your ChatGroq variable is named `model`.

llm_with_tools = model.bind_tools(tools)

SYSTEM_PROMPT = """
You are a helpful chatbot with access to tools.
For questions asking for current, latest, recent, live, today's, or otherwise
time-sensitive information, always use the DuckDuckGo search tool first.
Use the tool result to write a clear final answer. Do not mention internal
tool-call details unless they are relevant to the user's question.
When a tool result is already present in the conversation, answer directly
from that result and do not request another tool.
Use the stock tool for stock prices and the calculator for arithmetic.
"""


# -------------------
# 3. State
# -------------------

class ChatState(TypedDict):
    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]


# -------------------
# 4. Nodes
# -------------------

def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""

    messages = state["messages"]

    response = llm_with_tools.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), *messages]
    )

    return {
        "messages": [response]
    }


tool_node = ToolNode(tools)


def final_chat_node(state: ChatState):
    """Write the final answer after a tool result without requesting another tool."""

    response = model.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
    )

    return {
        "messages": [response]
    }


# -------------------
# 5. Checkpointer
# -------------------

conn = sqlite3.connect(
    database="chatbot.db",
    check_same_thread=False
)

checkpointer = SqliteSaver(conn=conn)


# -------------------
# 6. Graph
# -------------------

graph = StateGraph(ChatState)

graph.add_node(
    "chat_node",
    chat_node
)

graph.add_node(
    "tools",
    tool_node
)

graph.add_node(
    "final_chat_node",
    final_chat_node
)

graph.add_edge(
    START,
    "chat_node"
)

graph.add_conditional_edges(
    "chat_node",
    tools_condition
)

graph.add_edge(
    "tools",
    "final_chat_node"
)

graph.add_edge(
    "final_chat_node",
    END
)

chatbot = graph.compile(
    checkpointer=checkpointer
)


# -------------------
# 7. Helper
# -------------------

def retrieve_all_threads():

    all_threads = set()

    for checkpoint in checkpointer.list(None):

        all_threads.add(
            checkpoint.config[
                "configurable"
            ]["thread_id"]
        )

    return list(all_threads)