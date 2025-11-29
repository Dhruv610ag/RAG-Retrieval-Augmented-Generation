from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
from sklearn.metrics.pairwise import cosine_similarity
load_dotenv()

embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text="Delhi is the capital of India."
documents=["Delhi is the capital of India.",
           "Paris is the capital of France.",
           "London is the capital of UK."
           "kolkata is the capital of West Bengal."]

result=embeddings.embed_query(text)
#print("Embedding for the text is: ", str(result))
result_document=embeddings.embed_documents(documents)
#print("Embedding for the documents are: ", str(result_document))
similarity_score=cosine_similarity([result], result_document)
print("Similarity score is: ", str(similarity_score))

#retriever is like the search engine which will retrieve the most relevant documents based on the query
