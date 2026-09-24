# ResearchMate RAG Engine

ResearchMate is a FastAPI-powered document question-answering application. Upload PDF, DOCX, or TXT files, index their contents, and ask questions with cited responses.

## Features

- Upload and index PDF, DOCX, and TXT documents
- Semantic retrieval using OpenAI embeddings
- Answer generation using Anthropic Claude
- Streaming responses with source citations
- Optional web-search fallback through SerpAPI

## Requirements

- Python 3.10 or newer
- OpenAI API key
- Anthropic API key
- Optional SerpAPI key for web-search fallback

## Local setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
SERPAPI_API_KEY=your_serpapi_api_key
```

Start the application:

```powershell
uvicorn api_app:app --reload --port 5001
```

Open http://127.0.0.1:5001 in your browser.

## Deploying

GitHub stores the source code but does not run this Python backend through GitHub Pages. Deploy the repository as a web service on Render, Railway, or another Python hosting provider.

For Render, use:

```text
Build command: pip install -r requirements.txt
Start command: uvicorn api_app:app --host 0.0.0.0 --port $PORT
```

Add `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` as hosted-service environment variables. Add `SERPAPI_API_KEY` if web search is enabled.

## Security and storage

Never commit `.env`, API keys, uploaded documents, or generated Python cache files. These are excluded by `.gitignore`.

Uploaded documents and the in-memory index are not persistent across many hosted-service restarts or redeployments. Production deployments should use durable object storage and a persistent vector database.

## Project structure

- `api_app.py` - FastAPI routes and streaming API
- `rag.py` - document loading, embeddings, retrieval, and answer generation
- `index.html` - application interface
- `script.js` - browser interaction and streaming response handling
- `styles.css` - interface styling
- `requirements.txt` - Python dependencies
