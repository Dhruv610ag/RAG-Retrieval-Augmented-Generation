#helps in loading the csv file content 
from langchain_community.document_loaders import CSVLoader

loader=CSVLoader(file_path=r"C:\Users\DHRUV AGARWAL\Desktop\RAG-Retrieval-Augmented-Generation-\Social_Network_Ads.csv")

docs=loader.load()
print(docs[0].page_content)