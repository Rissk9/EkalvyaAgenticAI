"""
Singleton module – creates LLM, embeddings, and vectorstore ONCE at startup.
All other modules import from here.
"""
from functools import lru_cache

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
def get_vectorstore() -> FAISS:
    s = get_settings()
    loader = PyPDFLoader(s.resume_pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    return FAISS.from_documents(docs, get_embeddings())


@lru_cache
def get_retriever():
    return get_vectorstore().as_retriever(search_kwargs={"k": 3})
