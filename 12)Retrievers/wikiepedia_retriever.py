"""
In generative AI, a retriever is a component that fetches relevant documents from a vector store to provide context for a large language model (LLM). 
It works by converting a user's query into a vector and then performing a similarity search to find the most semantically similar vectors (and thus,
the most relevant documents) stored in the database. This process allows LLMs to generate more accurate and fact-based responses by grounding them in external data,
a technique known as Retrieval-Augmented Generation (RAG). 
"""

from langchain_community.retrievers import WikipediaRetriever
r=WikipediaRetriever(top_k_results=2,lang="en")

query="Artificial intelligence"

docs=r.invoke(query)
for i,doc in enumerate(docs):
    print(f"------result------:{i+1}")
    print("content:",doc.page_content)
    