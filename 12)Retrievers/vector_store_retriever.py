from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

documents=[
    Document(page_content="Langchain helps developers build LLM applications easily"),
    Document(page_content="Chroma is a vector database optimized for LLM based search"),
    Document(page_content="Embeddings convert text into high-dimensional vectors"),
    Document(page_content="OpenAI provides powerful embedding models")
]

embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store=Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="sample"
)

retriever=vector_store.as_retriever(search_kwards={"k":2})

query="What does embedding do?"

results=retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n--------------")
    print(doc.page_content)

print(vector_store.similarity_search_with_score(query, k=2))
