from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

load_dotenv()

# LLM Model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

video_id = "DlIAd4Rtkr8"

# ---------------------------
# 1. Fetch YouTube Transcript
# ---------------------------
try:
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
    transcript = " ".join(chunk["text"] for chunk in transcript_list)

except TranscriptsDisabled:
    print("No caption available for the video")

# --------------------------------------
# 2. Split transcript into text chunks
# --------------------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.create_documents([transcript])

# --------------------------------------
# 3. Embeddings (Correct Model Name!)
# --------------------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --------------------------------------
# 4. Create FAISS vector store (Correct!)
# --------------------------------------
vector_store = FAISS.from_documents(chunks, embedding_model)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)

# --------------------------------------
# 5. Prompt Template
# --------------------------------------
prompt = PromptTemplate(
    template=(
        "You are a helpful agent. Answer strictly from the provided transcript. "
        "If the context is insufficient, answer: 'I don't know'.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}"
    ),
    input_variables=["context", "question"]
)

# --------------------------------------
# 6. Get retrieved context
# --------------------------------------
question = "What is the video about?"
retrieved_docs = retriever.invoke(question)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

# --------------------------------------
# 7. Create final prompt for the model
# --------------------------------------
final_prompt_text = prompt.format(context=context_text, question=question)

# --------------------------------------
# 8. Get LLM answer
# --------------------------------------
answer = model.invoke(final_prompt_text)
print(answer.content)
