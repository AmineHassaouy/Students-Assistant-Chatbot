import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader

load_dotenv()

persist_dir = "db/chroma_db"


def load_documents():
    docs = []

    for file in os.listdir("docs"):
        if file.endswith(".txt"):
            path = os.path.join("docs", file)

            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Fix UTF-8 encoding issues
            content = content.encode("utf-8", "ignore").decode("utf-8")

            # Load metadata and replace content
            d = TextLoader(path).load()[0]
            d.page_content = content
            docs.append(d)

    print(f"Loaded {len(docs)} documents (UTF-8 preserved)")
    return docs


def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n"]
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")
    return chunks


def create_vectorstore(chunks):
    print("Embedding using SentenceTransformer model...")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    print(f"Vectorstore saved to {persist_dir}")


if __name__ == "__main__":
    # ⚠️ Clear old DB first if needed
    if os.path.exists(persist_dir):
        print("Removing old FAISS index...")
        import shutil
        shutil.rmtree(persist_dir)

    docs = load_documents()
    chunks = chunk_documents(docs)
    create_vectorstore(chunks)
