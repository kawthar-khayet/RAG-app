# app.py — version corrigée

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma          # ✅ corrigé

from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

app = FastAPI()

# Load vector DB and set up retriever
embedding_fn = OllamaEmbeddings(model="nomic-embed-text:latest")  # ✅ espaces retirés
vectordb = Chroma(persist_directory="vectordb", embedding_function=embedding_fn)
retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# LLM
llm = ChatOllama(model="llama3.1", temperature=0.0)          # ✅ nom du modèle corrigé
prompt = PromptTemplate.from_template(
    "Use the following context to answer the question. "
    "If the answer is not in the context, say 'I don't know.'"
    "\n\nContext:\n{context}\n\nQuestion: {question}"
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

class QueryIn(BaseModel):
    question: str

@app.post("/ask")
def ask(q: QueryIn):
    result = qa({"query": q.question})
    return {
        "answer": result["result"],
        "sources": [doc.metadata for doc in result.get("source_documents", [])]
    }