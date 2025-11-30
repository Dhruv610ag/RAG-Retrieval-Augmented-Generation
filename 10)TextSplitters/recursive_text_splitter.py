from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

loader=TextLoader(r"C:\Users\DHRUV AGARWAL\Desktop\RAG-Retrieval-Augmented-Generation-\docs.txt")

docs=loader.load()

splitter=RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

result=splitter.split_documents(docs)
print(len(result))