from rag.loader import load_documents
from rag.chunker import chunk_documents
from rag.embeddings import get_embedding_model
from rag.vector_store import create_vector_store, load_vector_store
from rag.retriever import get_retriever
from rag.generator import generate_answer

# 1. Load documents
docs = load_documents("data/documents")

# 2. Chunk
chunks = chunk_documents(docs)

# 3. Embeddings
embedding_model = get_embedding_model()

# 4. Vector store
vectorstore = create_vector_store(chunks, embedding_model)

# 5. Retriever
retriever = get_retriever(vectorstore)

# 6. Ask question
query = "What does the document say about free tier pricing?"
retrieved_docs = retriever.get_relevant_documents(query)

answer, sources = generate_answer(query, retrieved_docs)

print("\nANSWER:\n", answer)
print("\nSOURCES:")
for s in sources:
    print("-", s.metadata)
