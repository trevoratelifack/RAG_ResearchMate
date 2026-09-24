import asyncio
import math
import os
from pathlib import Path
from pypdf import PdfReader
from docx import Document
import re
from anthropic import AsyncAnthropic
from openai import OpenAI
import requests


def load_env_file():
    env_path = Path(__file__).with_name(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


load_env_file()

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("Missing OPENAI_API_KEY. Set it in .env or your shell before running rag.py.")

if not os.getenv("ANTHROPIC_API_KEY"):
    raise RuntimeError("Missing ANTHROPIC_API_KEY. Set it in .env or your shell before running rag.py.")

anthropic_client = AsyncAnthropic()
openai_client = OpenAI()

MODEL = "claude-sonnet-4-6"
EMBEDDING_MODEL = "text-embedding-3-small"


def read_pdf(path):
    """Extract text from a PDF file."""
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

def read_word(path):
    """Extract text from a Word document."""
    doc = Document(path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text.strip()

def read_text(path):
    """Extract text from a plain text file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def clean(text):
    """Clean up text by removing extra whitespace and newlines."""
    text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with single space
    return text.strip()

def load(path):
    """Load notes from a file, depending on its extension."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return clean(read_pdf(path))
    elif ext in [".docx", ".doc"]:
        return clean(read_word(path))
    elif ext == ".txt":
        return clean(read_text(path))
    else:
        raise ValueError(f"Unsupported file type: {ext}")



CHUNK_SIZE = 80
CHUNK_OVERLAP = 20
NOTE_CHUNKS = []

# Start with no notes — users can import files at runtime using the `import` command.
NOTES = []


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split a text into overlapping chunks of words."""
    words = text.split()
    if not words:
        return [""]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------
# STEP 0 - the Lesson 7 machinery, unchanged.
# ---------------------------------------------------------------

def embed(texts):
    """Turn a list of strings into a list of number-lists."""
    response = openai_client.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL,
    )
    return [item.embedding for item in response.data]


def cosine(a, b):
    """One score for how close two embeddings are. 1 = same, 0 = unrelated."""
    dot = 0.0
    size_a = 0.0
    size_b = 0.0
    for i in range(len(a)):
        dot = dot + a[i] * b[i]
        size_a = size_a + a[i] * a[i]
        size_b = size_b + b[i] * b[i]
    length = math.sqrt(size_a) * math.sqrt(size_b)
    if length == 0:
        return 0.0
    return dot / length


def prepare_notes():
    """Embed every note chunk once, at startup, and attach the vector to the chunk."""
    NOTE_CHUNKS.clear()
    note_texts = []
    for note in NOTES:
        chunks = chunk_text(note["content"])
        for chunk_index, chunk in enumerate(chunks, start=1):
            NOTE_CHUNKS.append(
                {
                    "note_id": note["id"],
                    "chunk_index": chunk_index,
                    "content": chunk,
                }
            )
            note_texts.append(chunk)

    if not note_texts:
        # Nothing to embed yet.
        return

    note_vectors = embed(note_texts)
    for i in range(len(NOTE_CHUNKS)):
        NOTE_CHUNKS[i]["vector"] = note_vectors[i]


# ---------------------------------------------------------------
# STEP 1 - RETRIEVE. Find the notes closest to the question.
# ---------------------------------------------------------------

def retrieve(query, how_many=2):
    """Return the note chunks whose meaning is closest to the query."""
    query_vector = embed([query])[0]

    pool = []
    for chunk in NOTE_CHUNKS:
        score = cosine(query_vector, chunk["vector"])
        pool.append({"score": score, "chunk": chunk})

    winners = []
    for _ in range(how_many):
        if len(pool) == 0:
            break
        best = 0
        for i in range(len(pool)):
            if pool[i]["score"] > pool[best]["score"]:
                best = i
        winners.append(pool[best])
        pool.pop(best)

    return winners


def import_files(paths_str: str):
    """Import one or more files specified by a comma-separated paths string.
    Returns the number of successfully imported files."""
    paths = [p.strip().strip('"').strip("'") for p in paths_str.split(",") if p.strip()]
    if not paths:
        print("No paths provided to import.")
        return 0

    added = 0
    next_id = max((n.get("id", 0) for n in NOTES), default=0) + 1
    for p in paths:
        ppath = Path(p)
        if not ppath.exists():
            print(f"File not found: {p}")
            continue
        try:
            content = load(ppath)
        except Exception as e:
            print(f"Failed to load {p}: {e}")
            continue
        NOTES.append({"id": next_id, "content": content, "source": str(ppath.name)})
        next_id += 1
        added += 1

    if added:
        print(f"Imported {added} file(s). Re-embedding notes...")
        prepare_notes()
    else:
        print("No files were imported.")

    return added


# ---------------------------------------------------------------
# STEP 2 - AUGMENT. Put the retrieved notes into the prompt.
# ---------------------------------------------------------------

def build_context(winners):
    """Turn the retrieved note chunks into one numbered block of text."""
    lines = []
    for i, winner in enumerate(winners, start=1):
        chunk = winner["chunk"]
        lines.append(f"[{i}] {chunk['content']}")
    return "\n\n".join(lines)


def build_prompt(question, context):
    """The grounding instruction is what makes this RAG and not just chat."""
    return (
        "Answer the question using ONLY the numbered notes below.\n"
        "If the notes do not contain the answer, say: "
        '"My notes don\'t cover that."\n'
        "Cite every factual statement with its supporting note number in square brackets, "
        "for example: 'The service retries failed jobs. [1]'. "
        "Use only citations that correspond to the notes provided.\n\n"
        "NOTES:\n"
        + context
        + "\n\nQUESTION: "
        + question
    )


# ---------------------------------------------------------------
# STEP 3 - GENERATE. Ask Claude, with the notes in front of it.
# ---------------------------------------------------------------

async def generate(prompt=None, messages=None):
    """Send messages or a single prompt to Claude and return the text answer.

    If `messages` is provided, it will be passed directly. Otherwise `prompt` is used
    as a single user message.
    """
    if messages is None:
        messages = [{"role": "user", "content": prompt}]

    response = await anthropic_client.messages.create(
        model=MODEL,
        max_tokens=500,
        messages=messages,
    )
    return response.content[0].text


async def stream_generate(messages):
    """Yield Claude's visible response text as it is generated."""
    async with anthropic_client.messages.stream(
        model=MODEL,
        max_tokens=500,
        messages=messages,
    ) as stream:
        async for text in stream.text_stream:
            yield text


async def ask_without_notes(question, history=None):
    """The 'before' case: Claude answering from memory alone, with optional chat history."""
    messages = []
    if history:
        # assume history is a list of {'role':..., 'content':...}
        messages.extend(history)
    messages.append({"role": "user", "content": question})

    response = await anthropic_client.messages.create(
        model=MODEL,
        max_tokens=500,
        messages=messages,
    )
    return response.content[0].text


# ---------------------------------------------------------------
# The whole pipeline in one function.
# ---------------------------------------------------------------

def web_search(query, max_items=5):
    """Perform a web search using SerpAPI, Bing (if keys provided), or DuckDuckGo as fallback.

    Returns a list of dicts: {'title', 'snippet', 'url'}
    """
    # 1) SerpAPI
    serpapi_key = os.getenv("SERPAPI_API_KEY")
    if serpapi_key:
        try:
            # Use the official serpapi Python client when available.
            try:
                from serpapi import Client as SerpClient
            except Exception:
                SerpClient = None

            if SerpClient:
                client = SerpClient(api_key=serpapi_key)
                params = {
                    "engine": "google",
                    "q": query,
                    "google_domain": "google.com",
                    "hl": "en",
                    "gl": "us",
                }
                data = client.search(params)
            else:
                # Fallback to direct HTTP call if client is not installed
                params = {"q": query, "api_key": serpapi_key, "engine": "google", "num": max_items}
                res = requests.get("https://serpapi.com/search.json", params=params, timeout=10)
                data = res.json()

            items = data.get("organic_results") or data.get("organic") or data.get("results") or []
            results = []
            for it in items[:max_items]:
                results.append({
                    "title": it.get("title") or it.get("position") or it.get("name") or "Result",
                    "snippet": it.get("snippet") or it.get("snippet_text") or it.get("snippet_html") or it.get("description") or "",
                    "url": it.get("link") or it.get("url") or it.get("displayed_link") or "",
                })
            return results
        except Exception:
            pass

    # 2) Bing Web Search
    bing_key = os.getenv("BING_API_KEY")
    if bing_key:
        try:
            headers = {"Ocp-Apim-Subscription-Key": bing_key}
            params = {"q": query, "count": max_items}
            res = requests.get("https://api.bing.microsoft.com/v7.0/search", headers=headers, params=params, timeout=10)
            data = res.json()
            webpages = (data.get("webPages") or {}).get("value", [])
            results = []
            for it in webpages[:max_items]:
                results.append({"title": it.get("name"), "snippet": it.get("snippet"), "url": it.get("url")})
            return results
        except Exception:
            pass

    # 3) DuckDuckGo Instant Answer (fallback)
    try:
        res = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=8,
        )
        data = res.json()
    except Exception:
        return []

    results = []
    abstract = data.get("AbstractText") or ""
    abstract_url = data.get("AbstractURL") or ""
    if abstract:
        results.append({"title": "Abstract", "snippet": abstract, "url": abstract_url})

    related = data.get("RelatedTopics") or []
    for item in related:
        if len(results) >= max_items:
            break
        if isinstance(item, dict):
            text = item.get("Text") or ""
            url = item.get("FirstURL") or ""
            if text:
                results.append({"title": text.split(" - ")[0], "snippet": text, "url": url})

    return results


async def answer(question, history=None):
    winners = retrieve(question)
    context = build_context(winners)
    prompt = build_prompt(question, context)

    # Determine if retrieval produced a sufficiently relevant result
    best_score = 0.0
    if winners:
        best_score = float(winners[0]["score"])

    # If no winners or low relevance, perform web search fallback before calling model
    RELEVANCE_THRESHOLD = float(os.getenv("RAG_RELEVANCE_THRESHOLD", "0.2"))
    if not winners or best_score < RELEVANCE_THRESHOLD:
        web_results = web_search(question)
        if web_results:
            web_context_lines = []
            for i, r in enumerate(web_results, start=1):
                web_context_lines.append(f"[Web {i}] {r.get('title','')} - {r.get('snippet','')} (URL: {r.get('url','')})")
            web_context = "\n\n".join(web_context_lines)

            web_prompt = (
                "The notes are insufficient to answer. Use ONLY the following numbered web search results to answer the question. "
                "Be concise and clearly state that this answer is based on a web search. Cite each factual statement using "
                "the matching source number in square brackets, for example [1].\n\n"
                "WEB RESULTS:\n"
                + web_context
                + "\n\nQUESTION: "
                + question
            )

            messages = []
            if history:
                messages.extend(history)
            messages.append({"role": "user", "content": web_prompt})

            reply = await generate(messages=messages)
            web_sources = [{"source": r.get("url") or r.get("title"), "snippet": r.get("snippet")} for r in web_results]
            return reply, [], web_sources

    # Otherwise call model with RAG context
    messages = []
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    reply = await generate(messages=messages)

    # If model indicates notes don't cover it, do a web search fallback
    if isinstance(reply, str) and "My notes don't cover that." in reply:
        web_results = web_search(question)
        if web_results:
            web_context_lines = []
            for i, r in enumerate(web_results, start=1):
                web_context_lines.append(f"[Web {i}] {r['title']} - {r['snippet']} (URL: {r['url']})")
            web_context = "\n\n".join(web_context_lines)

            web_prompt = (
                "The notes did not contain the answer. Use ONLY the following numbered web search results to answer the question. "
                "Be concise and clearly state that this answer is based on a web search. Cite each factual statement using "
                "the matching source number in square brackets, for example [1].\n\n"
                "WEB RESULTS:\n"
                + web_context
                + "\n\nQUESTION: "
                + question
            )

            messages = []
            if history:
                messages.extend(history)
            messages.append({"role": "user", "content": web_prompt})

            reply = await generate(messages=messages)
            web_sources = [{"source": r.get("url") or r.get("title"), "snippet": r.get("snippet")} for r in web_results]
            return reply, [], web_sources

    return reply, winners, []


async def main():
    print("Embedding notes...")
    prepare_notes()
    print("Ready!\n")

    while True:
        try:
            question = input("\nAsk a question (or type 'quit' to exit): ").strip()
            if question.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            # Support importing files during the interactive session.
            if question.lower().startswith("import ") or question.lower().startswith("upload "):
                paths_part = question.split(" ", 1)[1]
                import_files(paths_part)
                continue
            if not question:
                continue

            print("\nChoose how to answer:")
            print("1. WITH RAG (uses notes)")
            print("2. WITHOUT RAG (Claude's memory only)")
            print("(Tip: import files using: import C:\\path\\to\\file.pdf or import file1.pdf,file2.docx")
            choice = input("Enter 1 or 2: ").strip()

            if choice == "1":
                print("\n" + "=" * 60)
                print("WITH RAG - Claude answers from our notes")
                print("=" * 60)
                reply, winners = await answer(question)
                print("Retrieved:")
                for winner in winners:
                    score = winner["score"]
                    print(
                        f" note {winner['chunk']['note_id']} chunk {winner['chunk']['chunk_index']} score {score:.3f}"
                    )
                print("\nAnswer:")
                print(reply)

            elif choice == "2":
                print("\n" + "=" * 60)
                print("WITHOUT RAG - Claude answers from memory")
                print("=" * 60)
                memory_answer = await ask_without_notes(question)
                print(memory_answer)

            else:
                print("Invalid choice. Please enter 1 or 2.")

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    asyncio.run(main())
