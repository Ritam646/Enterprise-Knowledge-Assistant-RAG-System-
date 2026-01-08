from langchain_community.vectorstores import FAISS
import os

INDEX_PATH = "data/index"

def create_vector_store(chunks, embedding_model):
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local(INDEX_PATH)
    return vectorstore


def load_vector_store(embedding_model):
    return FAISS.load_local(
        INDEX_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
