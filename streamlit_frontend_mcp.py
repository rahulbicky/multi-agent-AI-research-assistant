import queue
import uuid

import streamlit as st

from langgraph_mcp_backend import (
    chatbot,
    retrieve_all_threads,
    submit_async_task
)

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage
)


# ============================================================
# 1. Utilities
# ============================================================

def generate_thread_id():
    return str(uuid.uuid4())


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat():
    thread_id = generate_thread_id()

    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []

    add_thread(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    return state.values.get("messages", [])


# ============================================================
# 2. Session Initialization
# ============================================================

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []


if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()


if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()


add_thread(
    st.session_state["thread_id"]
)


# ============================================================
# 3. Sidebar
# ============================================================

st.sidebar.title("🤖 LangGraph MCP Chatbot")


if st.sidebar.button("➕ New Chat"):
    reset_chat()
    st.rerun()


st.sidebar.header("💬 My Conversations")


for thread_id in reversed(
    st.session_state["chat_threads"]
):

    if st.sidebar.button(
        str(thread_id),
        key=f"thread_{thread_id}"
    ):

        st.session_state["thread_id"] = thread_id

        messages = load_conversation(thread_id)

        temp_messages = []

        for msg in messages:

            # Only show Human and AI messages
            if isinstance(msg, HumanMessage):

                temp_messages.append({
                    "role": "user",
                    "content": msg.content
                })

            elif isinstance(msg, AIMessage):

                if msg.content:

                    temp_messages.append({
                        "role": "assistant",
                        "content": msg.content
                    })

        st.session_state["message_history"] = temp_messages

        st.rerun()


# ============================================================
# 4. Display Previous Messages
# ============================================================

for message in st.session_state["message_history"]:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# 5. User Input
# ============================================================

user_input = st.chat_input(
    "Ask me anything..."
)


if user_input:

    # --------------------------------------------------------
    # Display User Message
    # --------------------------------------------------------

    st.session_state["message_history"].append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):

        st.markdown(user_input)


    # --------------------------------------------------------
    # LangGraph Configuration
    # --------------------------------------------------------

    CONFIG = {

        "configurable": {
            "thread_id": st.session_state["thread_id"]
        },

        "metadata": {
            "thread_id": st.session_state["thread_id"]
        },

        "run_name": "chat_turn"
    }


    # --------------------------------------------------------
    # Assistant Response
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        status_holder = {
            "box": None
        }


        def ai_only_stream():

            event_queue = queue.Queue()


            async def run_stream():

                try:

                    async for (
                        message_chunk,
                        metadata
                    ) in chatbot.astream(
                        {
                            "messages": [
                                HumanMessage(
                                    content=user_input
                                )
                            ]
                        },

                        config=CONFIG,

                        stream_mode="messages"
                    ):

                        event_queue.put(
                            (
                                message_chunk,
                                metadata
                            )
                        )


                except Exception as exc:

                    event_queue.put(
                        (
                            "error",
                            exc
                        )
                    )


                finally:

                    event_queue.put(None)


            # Start async LangGraph execution
            submit_async_task(
                run_stream()
            )


            # ------------------------------------------------
            # Read events
            # ------------------------------------------------

            while True:

                item = event_queue.get()


                if item is None:
                    break


                message_chunk, metadata = item


                # --------------------------------------------
                # Error
                # --------------------------------------------

                if message_chunk == "error":

                    raise metadata


                # --------------------------------------------
                # Tool Execution
                # --------------------------------------------

                if isinstance(
                    message_chunk,
                    ToolMessage
                ):

                    tool_name = getattr(
                        message_chunk,
                        "name",
                        "tool"
                    )


                    if status_holder["box"] is None:

                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}`...",
                            expanded=True
                        )

                    else:

                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}`...",
                            state="running",
                            expanded=True
                        )


                # --------------------------------------------
                # AI Response
                # --------------------------------------------

                if isinstance(
                    message_chunk,
                    AIMessage
                ):

                    if message_chunk.content:

                        yield message_chunk.content


        # ----------------------------------------------------
        # Stream response in UI
        # ----------------------------------------------------

        ai_message = st.write_stream(
            ai_only_stream()
        )


        # ----------------------------------------------------
        # Tool Finished
        # ----------------------------------------------------

        if status_holder["box"] is not None:

            status_holder["box"].update(
                label="✅ Tool finished",
                state="complete",
                expanded=False
            )


    # ========================================================
    # 6. Save Assistant Message
    # ========================================================

    st.session_state["message_history"].append(
        {
            "role": "assistant",
            "content": ai_message
        }
    )