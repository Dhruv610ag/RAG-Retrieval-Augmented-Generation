#This loader helps in loading the simple txt file 
from langchain_community.document_loaders import TextLoader

loader=TextLoader(r"C:\Users\DHRUV AGARWAL\Desktop\RAG-Retrieval-Augmented-Generation-\docs.txt",encoding="utf-8")

docs=loader.load()

print(docs)
print(type(docs))
print(docs[0].page_content)
print(docs[0].metadata)