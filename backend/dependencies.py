"""
Singleton module – creates LLM, embeddings, and vectorstore ONCE at startup.
All other modules import from here.

The vectorstore and retriever are optional — if no resume PDF is found
or embeddings fail, the app still starts and works (just without RAG context).
"""
import os
from functools import lru_cache
from typing import Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from backend.config import get_settings


@lru_cache
def get_llm() -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(
        base_url=s.llm_base_url,
        api_key=s.llm_api_key,
        model=s.llm_model,
        temperature=s.llm_temperature,
    )


@lru_cache
def get_embeddings() -> OpenAIEmbeddings:
    s = get_settings()
    return OpenAIEmbeddings(
        model=s.embedding_model,
        base_url=s.embedding_base_url,
        api_key=s.embedding_api_key,
    )


@lru_cache
def get_vectorstore() -> Optional[FAISS]:
    """Build FAISS vectorstore from resume PDF. Returns None if unavailable."""
    s = get_settings()
    pdf_path = s.resume_pdf_path

    if not pdf_path:
        print("[INFO] No resume configured (RESUME_PDF_PATH not set) -- running without RAG context")
        return None

    if not os.path.exists(pdf_path):
        print(f"[WARN] Resume PDF not found at '{pdf_path}' -- running without RAG context")
        return None

    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        if not documents:
            print("[WARN] Resume PDF loaded but contained no pages -- running without RAG context")
            return None

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_documents(documents)

        vectorstore = FAISS.from_documents(docs, get_embeddings())
        print(f"[OK] Vectorstore built from '{pdf_path}' ({len(docs)} chunks)")
        return vectorstore

    except Exception as e:
        print(f"[WARN] Failed to build vectorstore: {e} -- running without RAG context")
        return None


@lru_cache
def get_retriever():
    """Return retriever if vectorstore is available, otherwise None."""
    vs = get_vectorstore()
    if vs is None:
        return None
    return vs.as_retriever(search_kwargs={"k": 3})
