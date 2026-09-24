from pathlib import Path

path = Path(r'c:\Users\claud\OneDrive\Desktop\Python code\RAG\Rags')
code = r'''import asyncio
import math
import os
from pathlib import Path

from anthropic import AsyncAnthropic
from openai import OpenAI


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
    raise RuntimeError(
        "Missing OPENAI_API_KEY. Set it in .env or your shell before running rag.py."
    )

if not os.getenv("ANTHROPIC_API_KEY"):
    raise RuntimeError(
        "Missing ANTHROPIC_API_KEY. Set it in .env or your shell before running rag.py."
    )

anthropic_client = AsyncAnthropic()
openai_client = OpenAI()

MODEL = "claude-sonnet-4-6"
EMBEDDING_MODEL = "text-embedding-3-small"


# ---------------------------------------------------------------
# Our notes. In a real project these come from files or a database.
# ---------------------------------------------------------------

NOTES = [
    {
        "id": 1,
        "content": (
            "Our team decided to store embeddings in a plain Python list "
            "for now. We will move to a real vector database only when we "
            "pass 50,000 notes."
        ),
    },
    {
        "id": 2,
        "content": (
            "An embedding turns text into numbers so that similar texts end "
            "up close together. We use voyage-3.5 because it is the provider "
            "Anthropic recommends."
        ),
    },
    {
        "id": 3,
        "content": (
            "The agent loop keeps calling tools until Claude stops asking for "
            "them. We cap it at 10 steps so a confused agent cannot run "
            "forever."
        ),
    },
    {
        "id": 4,
        "content": (
            "Standup moved to Tuesdays at 9am. Friday demos are cancelled "
            "until further notice."
        ),
    },
]


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
        dot += a[i] * b[i]
        size_a += a[i] * a[i]
        size_b += b[i] * b[i]
    length = math.sqrt(size_a) * math.sqrt(size_b)
    if length == 0:
        return 0.0
    return dot / length


def prepare_notes():
    """Embed every note once, at startup, and attach the vector to the note."""
    note_texts = [note["content"] for note in NOTES]
    note_vectors = embed(note_texts)
    for note, vector in zip(NOTES, note_vectors):
        note["vector"] = vector


# ---------------------------------------------------------------
# STEP 1 - RETRIEVE. Find the notes closest to the question.
# ---------------------------------------------------------------


def retrieve(query, how_many=2):
    """Return the notes whose meaning is closest to the query."""
    query_vector = embed([query])[0]

    pool = []
    for note in NOTES:
        score = cosine(query_vector, note["vector"])
        pool.append({"score": score, "note": note})

    # Find the biggest, take it out, repeat. No sort key, no lambda.
    winners = []
    for _ in range(how_many):
        if not pool:
            break
        best = 0
        for i in range(len(pool)):
            if pool[i]["score"] > pool[best]["score"]:
                best = i
        winners.append(pool.pop(best))

    return winners


# ---------------------------------------------------------------
# STEP 2 - AUGMENT. Put the retrieved notes into the prompt.
# ---------------------------------------------------------------


def build_context(winners):
    """Turn the retrieved notes into one numbered block of text."""
    lines = []
    for i, winner in enumerate(winners, start=1):
        note = winner["note"]
        lines.append(f"[Note {i}] {note['content']}")
    return "\n\n".join(lines)


def build_prompt(question, context):
    """The grounding instruction is what makes this RAG and not just chat."""
    return (
        "Answer the question using ONLY the notes below.\n"
        "If the notes do not contain the answer, say: "
        "\"My notes don't cover that.\"\n"
        "End your answer with the note number you used, like [Note 1].\n\n"
        "NOTES:\n"
        + context
        + "\n\nQUESTION: "
        + question
    )


# ---------------------------------------------------------------
# STEP 3 - GENERATE. Ask Claude, with the notes in front of it.
# ---------------------------------------------------------------


async def generate(prompt):
    """Send the augmented prompt to Claude and return the text answer."""
    response = await anthropic_client.messages.create(
        model=MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


async def ask_without_notes(question):
    """The 'before' case: Claude answering from memory alone."""
    response = await anthropic_client.messages.create(
        model=MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": question}],
    )
    return response.content[0].text


# ---------------------------------------------------------------
# The whole pipeline in one function.
# ---------------------------------------------------------------


async def answer(question):
    winners = retrieve(question)
    context = build_context(winners)
    prompt = build_prompt(question, context)
    reply = await generate(prompt)
    return reply, winners


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
            if not question:
                continue

            print("\nChoose how to answer:")
            print("1. WITH RAG (uses notes)")
            print("2. WITHOUT RAG (Claude's memory only)")
            choice = input("Enter 1 or 2: ").strip()

            if choice == "1":
                print("\n" + "=" * 60)
                print("WITH RAG - Claude answers from our notes")
                print("=" * 60)
                reply, winners = await answer(question)
                print("Retrieved:")
                for winner in winners:
                    score = winner["score"]
                    note_id = winner["note"]["id"]
                    print(f" note {note_id} score {score:.3f}")
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
'''

path.write_text(code, encoding='utf-8')
print('rewritten', path)
