from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

# Load documents
loader = TextLoader(r"C:\Users\DHRUV AGARWAL\Desktop\RAG-Retrieval-Augmented-Generation-\docs.txt")
documents = loader.load()

# Split text
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Vector store
vectorstore = FAISS.from_documents(docs, embeddings)

# Retriever
retriever = vectorstore.as_retriever()

# LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Prompt template
prompt = ChatPromptTemplate.from_template(
    """
    You are an expert AI.
    Use the context below to answer the question.

    Context:
    {context}

    Question:
    {question}
    """
)

# Build LCEL RAG chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# Query
query = "What are the various takeaways from the documents?"
response = rag_chain.invoke(query)

print("\nAnswer:\n", response.content)

"""
In the context of building AI application chains (specifically within frameworks like LangChain),
a RunnablePassthrough is a component that passes the input data 
through to the next step in the chain without any modification
"""