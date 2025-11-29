from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

loader=TextLoader(r"C:\Users\DHRUV AGARWAL\Desktop\RAG-Retrieval-Augmented-Generation-\docs.txt")

docs=loader.load()

splitter=CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separator=""
)

result=splitter.split_documents(docs)

print(result[1].page_content)