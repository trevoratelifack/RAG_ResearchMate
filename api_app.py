import json
import os
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

from rag import NOTES, MODEL, answer, anthropic_client, build_context, build_prompt, import_files, retrieve, stream_generate

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="ResearchMate Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def read_root():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/styles.css")
async def read_styles():
    return FileResponse(BASE_DIR / "styles.css")


@app.get("/script.js")
async def read_script():
    return FileResponse(BASE_DIR / "script.js")


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected.")

    allowed = {"pdf", "doc", "docx", "txt"}
    ext = Path(file.filename).suffix.lower().lstrip(".")
    if ext not in allowed:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    safe_name = os.path.basename(file.filename)
    target = UPLOAD_DIR / safe_name
    counter = 1
    while target.exists():
        target = UPLOAD_DIR / f"{Path(safe_name).stem}_{counter}{Path(safe_name).suffix}"
        counter += 1

    contents = await file.read()
    target.write_bytes(contents)

    try:
        imported = import_files(str(target))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not process file: {exc}") from exc

    if imported == 0:
        raise HTTPException(status_code=400, detail="The file type is not supported or could not be read.")

    return {"success": True, "message": "File uploaded and indexed.", "filename": target.name, "notes_count": len(NOTES)}


class Message(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    question: str
    history: List[Message] = []


def build_sources(winners):
    """Create the ordered source records used by inline [1], [2] citations."""
    sources = []
    for winner in winners:
        chunk = winner["chunk"]
        note = next((n for n in NOTES if n.get("id") == chunk["note_id"]), None)
        sources.append({
            "note_id": chunk["note_id"],
            "chunk_index": chunk["chunk_index"],
            "score": round(float(winner["score"]), 4),
            "source": note.get("source") if note else "Unknown source",
        })
    return sources


@app.post("/api/ask/stream")
async def ask_question_stream(payload: AskRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    async def events():
        def event(kind, data):
            return f"event: {kind}\ndata: {json.dumps(data)}\n\n"

        try:
            if NOTES:
                yield event("status", "Thinking · searching indexed documents…")
                winners = retrieve(question)
                sources = build_sources(winners)
                yield event("sources", sources)
                yield event("status", "Thinking · drafting a cited response…")
                messages = [m.dict() for m in payload.history]
                messages.append({"role": "user", "content": build_prompt(question, build_context(winners))})
                async for token in stream_generate(messages):
                    yield event("token", token)
            else:
                yield event("status", "Thinking · drafting a response…")
                messages = [m.dict() for m in payload.history]
                messages.append({"role": "user", "content": question})
                async with anthropic_client.messages.stream(model=MODEL, max_tokens=500, messages=messages) as stream:
                    async for token in stream.text_stream:
                        yield event("token", token)
            yield event("done", None)
        except Exception as exc:
            yield event("error", str(exc))

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/ask")
async def ask_question(payload: AskRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    try:
        if NOTES:
            # pass conversation history to the RAG pipeline
            reply, winners, web_sources = await answer(question, history=[m.dict() for m in payload.history])
            sources = []
            if winners:
                sources = build_sources(winners)
                return {"success": True, "answer": reply, "sources": sources, "web_search": True}
            elif web_sources:
                # reply came from web search fallback
                sources = [{"source": w.get("source"), "snippet": w.get("snippet")} for w in web_sources]
                return {"success": True, "answer": reply, "sources": sources, "web_search": True}
            else:
                # no winners and no web results — return model reply without sources
                return {"success": True, "answer": reply, "sources": [], "web_search": True}
        else:
            # No notes uploaded yet: answer from the model's memory only
            reply = await ask_without_notes(question, history=[m.dict() for m in payload.history])
            return {"success": True, "answer": reply, "sources": [], "web_search": True}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_app:app", host="0.0.0.0", port=5001, reload=True)
