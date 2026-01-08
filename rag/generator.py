from rag.llm import get_llm

llm = get_llm()

def generate_answer(query, retrieved_docs):
    context = "\n\n".join(
        [doc.page_content for doc in retrieved_docs]
    )

    prompt = f"""
You are a document-grounded AI assistant.

Answer the question ONLY using the context below.
If the answer is not present, say "Not found in document".

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content, retrieved_docs
