# this type of loader helps in loading the online,webpages content

from langchain_community.document_loaders import WebBaseLoader

url="https://en.wikipedia.org/wiki/Earthquake"

loader=WebBaseLoader(url)

docs=loader.load()

print(docs)