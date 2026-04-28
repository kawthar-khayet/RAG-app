# embeddings.py — contenu complet

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import json, os


def embed_and_store(
    chunks_dir="chunks",
    persist_directory="vectordb",
    embedding_model="nomic-embed-text:latest"
):
    """Produit les embeddings de tous les chunks et les persiste dans Chroma."""
    embedder = OllamaEmbeddings(model=embedding_model)
    documents = []

    for fn in os.listdir(chunks_dir):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(chunks_dir, fn), "r", encoding="utf8") as f:
            items = json.load(f)
        for it in items:
            documents.append(
                Document(
                    page_content=it["page_content"],
                    metadata=it.get("metadata", {})
                )
            )

    print(f"Nombre total de documents à encoder : {len(documents)}")
    vectordb = Chroma.from_documents(
        documents,
        embedding=embedder,
        persist_directory=persist_directory
    )
    vectordb.persist()
    print(f"✅ Vector DB persisté dans '{persist_directory}'")
    return vectordb


# Lancer la création des embeddings
vectordb = embed_and_store()