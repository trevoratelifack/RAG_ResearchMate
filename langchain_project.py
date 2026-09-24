import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import zipfile

path = Path(r"C:\Users\claud\OneDrive\Desktop\Python code\RAG\RESUME.docx")

print("Exists:", path.exists())
print("Size:", path.stat().st_size if path.exists() else "N/A")
print("Is valid DOCX:", zipfile.is_zipfile(path))
# -------------------------------------------------------------------
# Step 1: load environment variables
# -------------------------------------------------------------------
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
        "Missing OPENAI_API_KEY. Add it to a .env file or your shell before running this app."
    )

# -------------------------------------------------------------------
# Step 2: helper to load documents from files
# -------------------------------------------------------------------
def load_documents_from_path(file_path: str) -> list[Document]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".txt":
        loader = TextLoader(str(path), encoding="utf-8")
        return loader.load()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
        return loader.load()

    if suffix in {".doc", ".docx"}:
        loader = UnstructuredWordDocumentLoader(str(path))
        return loader.load()

    raise ValueError(f"Unsupported file type: {suffix}. Use .txt, .pdf, .doc, or .docx")


# -------------------------------------------------------------------
# Step 3: create a small default knowledge base if no file is loaded
# -------------------------------------------------------------------
def build_default_knowledge_base() -> list[Document]:
    return [
        Document(
            page_content=(
                "LangChain is a framework for building applications with Large Language Models. "
                "It helps connect prompts, tools, memory, and retrieval to create AI workflows."
            ),
            metadata={"source": "intro"},
        ),
        Document(
            page_content=(
                "RAG stands for Retrieval-Augmented Generation. It combines a search system with a model so "
                "the model can answer using relevant documents instead of relying only on its training data."
            ),
            metadata={"source": "rag"},
        ),
        Document(
            page_content=(
                "A vector database stores embeddings, which are numerical representations of text. This makes it "
                "possible to search for semantically similar documents quickly."
            ),
            metadata={"source": "vector-db"},
        ),
    ]


# -------------------------------------------------------------------
# Step 4: build vectorstore from documents
# -------------------------------------------------------------------
def build_vectorstore(documents: list[Document]) -> FAISS:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_documents(chunks, embeddings)


knowledge_base = build_default_knowledge_base()
vectorstore = build_vectorstore(knowledge_base)

# -------------------------------------------------------------------
# Step 5: build the question-answering chain
# -------------------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant.
    Use only the context below to answer the user's question.
    If the answer is not in the context, say: I couldn't find that in the provided knowledge base.

    Context:
    {context}

    Question:
    {input}
    """
)

retriever = None
question_answer_chain = None


def rebuild_question_chain() -> None:
    global retriever, question_answer_chain
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    question_answer_chain = (
        {
            "context": retriever,
            "input": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


rebuild_question_chain()

# -------------------------------------------------------------------
# Step 6: define a function to ask questions
# -------------------------------------------------------------------
def ask_question(question: str) -> str:
    return question_answer_chain.invoke(question)


def load_file_into_memory(file_path: str) -> None:
    global vectorstore
    documents = load_documents_from_path(file_path)
    vectorstore = build_vectorstore(documents)
    rebuild_question_chain()
    print(f"Loaded and indexed: {file_path}")


# -------------------------------------------------------------------
# Step 7: simple terminal loop
# -------------------------------------------------------------------
def main() -> None:
    print("LangChain RAG Demo")

    # ---------------------------------------------------------------
    # Automatically load a document at startup
    # ---------------------------------------------------------------
    default_file = Path(__file__).parent / "RESUME.docx"

    if default_file.exists():
        try:
            load_file_into_memory(str(default_file))
            print(f"Startup document loaded: {default_file}")
        except Exception as exc:
            print(f"Error loading startup document: {exc}")
            print("Starting with the default knowledge base.")
    else:
        print(f"No startup document found: {default_file}")
        print("Starting with the default knowledge base.")

    print("\nType 'load <file>' to load another .txt, .pdf, .doc, or .docx file")
    print("Type 'quit' to exit")

    while True:
        try:
            user_input = input("\nAsk a question or command: ").strip()
        except EOFError:
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if user_input.lower().startswith("load "):
            file_path = user_input[5:].strip()

            # Remove surrounding quotes from Windows paths
            if (
                len(file_path) >= 2
                and file_path[0] == file_path[-1]
                and file_path[0] in {'"', "'"}
            ):
                file_path = file_path[1:-1]

            try:
                load_file_into_memory(file_path)
            except Exception as exc:
                print(f"Error loading file: {exc}")

            continue

        try:
            answer = ask_question(user_input)
            print("\nAnswer:")
            print(answer)
        except Exception as exc:
            print(f"Error answering question: {exc}")


if __name__ == "__main__":
    main()

