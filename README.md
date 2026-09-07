# Conversational AI Agent

A collection of practical conversational AI examples built with Python, LangChain, LangGraph, Streamlit, and Groq. The repository progresses from a basic chatbot to tool calling, persistent conversations, streaming responses, MCP integrations, and PDF-based RAG.

## What This Project Includes

| Workflow | Backend | Frontend | Main capability |
| --- | --- | --- | --- |
| Basic chatbot | `langgraph_backend.py` | `streamlit_frontend.py` | Minimal LangGraph + Groq chat |
| Streaming | `langgraph_backend.py` or streaming variant | `streamlit_frontend_streaming.py` | Token/message streaming |
| Threaded chat | `langgraph_backend.py` | `streamlit_frontend_threading.py` | Multiple conversation threads |
| Tools | `langgraph_tool_backend.py` | `streamlit_frontend.py` | Calculator, web search, stock lookup |
| Database memory | `langgraph_database_backend.py` | `streamlit_frontend_database.py` | SQLite-backed checkpoints |
| MCP | `langgraph_mcp_backend.py` | `streamlit_frontend_mcp.py` | Tools from an external MCP server |
| PDF RAG | `langraph_rag_backend.py` | `streamlit_rag_frontend.py` | Upload a PDF and ask questions about it |

The recommended demo is the **PDF RAG chatbot** because it combines Groq tool calling, local embeddings, FAISS retrieval, SQLite conversation memory, web search, calculator, and stock lookup.

## Architecture

```text
Streamlit frontend
	|
	v
LangGraph state graph
  chat node <--> tool node
	|
	+--> ChatGroq (Groq API)
	+--> FAISS + Hugging Face embeddings (local PDF retrieval)
	+--> DuckDuckGo search (no API key)
	+--> Alpha Vantage (optional stock API)
	+--> SQLite checkpoint database
```

For PDF RAG, the uploaded file is loaded with `PyPDFLoader`, split into chunks, embedded with `sentence-transformers/all-MiniLM-L6-v2`, and stored in a per-thread FAISS retriever. The embedding model runs locally after its first download; it does not require an API key.

## Requirements

- Windows, macOS, or Linux
- Python 3.10 or newer; Python 3.13 is supported by the current environment
- Internet access for Groq requests and the first Hugging Face model download
- A Groq API key for chat responses
- Optional Alpha Vantage API key for stock prices
- Optional LangSmith API key for tracing
- An MCP server at `http://127.0.0.1:8000/mcp` only for the MCP workflow

## Setup

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Configure environment variables

Copy `.env.example` to `.env` and fill in your own values:

```powershell
Copy-Item .env.example .env
```

Required:

| Variable | Required? | Used for |
| --- | --- | --- |
| `GROQ_API_KEY` | Yes | ChatGroq responses in all chatbot workflows |
| `ALPHAVANTAGE_API_KEY` | No | Stock-price tool only |
| `LANGSMITH_API_KEY` | No | LangSmith tracing |
| `LANGSMITH_PROJECT` | No | LangSmith project name |
| `LANGSMITH_ENDPOINT` | No | LangSmith endpoint; normally the default shown in `.env.example` |
| `LANGSMITH_TRACING` | No | Set to `true` only when tracing is wanted |
| `HF_TOKEN` | No | Only for higher Hugging Face download limits or private models |

Never commit `.env`, paste real keys into README files, or share keys in screenshots. If a key has already been exposed, revoke it and create a replacement.

## Run the Recommended PDF RAG App

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_rag_frontend.py
```

Then open the local URL shown by Streamlit, upload a PDF from the sidebar, wait for indexing to finish, and ask questions about the document. Start a new thread when you want an isolated conversation and document context.

The repository includes `.streamlit/config.toml`, which disables Streamlit's file watcher. This avoids noisy optional `transformers`/`torchvision` scan messages during development; `torchvision` is not required by this text-only RAG workflow.

## Run Other Workflows

Use the corresponding frontend as the Streamlit entry point:

```powershell
# Basic/tool chat
.\.venv\Scripts\python.exe -m streamlit run streamlit_frontend.py

# Streaming chat
.\.venv\Scripts\python.exe -m streamlit run streamlit_frontend_streaming.py

# Threaded chat
.\.venv\Scripts\python.exe -m streamlit run streamlit_frontend_threading.py

# SQLite/database example
.\.venv\Scripts\python.exe -m streamlit run streamlit_frontend_database.py

# MCP example; start the MCP server first at http://127.0.0.1:8000/mcp
.\.venv\Scripts\python.exe -m streamlit run streamlit_frontend_mcp.py
```

The MCP backend uses `langchain_mcp_adapters` and expects a streamable HTTP MCP server at:

```text
http://127.0.0.1:8000/mcp
```

Change that URL in `langgraph_mcp_backend.py` if your MCP server runs elsewhere.

## Tool Behavior

- **Calculator:** local arithmetic; no API key.
- **DuckDuckGo search:** web search through the community integration; no API key, but internet access is required.
- **Stock price:** calls `https://www.alphavantage.co/query` and requires `ALPHAVANTAGE_API_KEY`.
- **PDF RAG:** uses local Hugging Face embeddings and FAISS; no Hugging Face API key is required for the public model.
- **ChatGroq:** calls Groq using the configured `GROQ_API_KEY` and the `openai/gpt-oss-20b` model.

## Project Files

| File | Purpose |
| --- | --- |
| `langgraph_backend.py` | Basic Groq LangGraph backend |
| `langgraph_tool_backend.py` | Calculator, search, and stock tools |
| `langgraph_database_backend.py` | SQLite checkpoint example |
| `langgraph_mcp_backend.py` | External MCP tool integration |
| `langraph_rag_backend.py` | PDF ingestion, FAISS retrieval, and RAG tools |
| `streamlit_rag_frontend.py` | Recommended PDF chat UI |
| `requirements.txt` | Python dependencies with compatible LangChain pins |
| `.env.example` | Safe environment-variable template; contains no secrets |
| `.streamlit/config.toml` | Quiet Streamlit development configuration |

## Troubleshooting

### `ModuleNotFoundError: pypdf`

Install dependencies from the project environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### `ModuleNotFoundError: faiss`

The project uses the CPU package:

```powershell
.\.venv\Scripts\python.exe -m pip install faiss-cpu
```

### `langchain_core.messages.block_translators.langchain_v0` error

This indicates a mixed LangChain installation. Use the project interpreter and reinstall the pinned requirements:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

Stop the old Streamlit process completely before restarting it.

### Hugging Face warning

An unauthenticated Hugging Face Hub warning is normally harmless for the public embedding model. The model is cached locally after download. `HF_TOKEN` is optional.

### MCP tools are not loaded

Start the MCP server first and confirm that `http://127.0.0.1:8000/mcp` is reachable. The RAG workflow does not require an MCP server.

## Development Checks

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m py_compile langraph_rag_backend.py streamlit_rag_frontend.py
```

The runtime creates local SQLite files such as `chatbot.db`; these are intentionally ignored by Git because they contain user conversation state.

## Security Notes

- Keep all credentials in `.env` or a secret manager.
- Do not commit `.env`, database files, model caches, or API keys.
- Rotate any credential that has appeared in a public repository, screenshot, terminal log, or chat.
- The Alpha Vantage key is read from `ALPHAVANTAGE_API_KEY`; no key is hardcoded in the RAG backend.

## License

See [LICENSE](LICENSE).
