from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
prompt = "What is the capital of India? Also describe places to hang around with friends."
result = model.invoke(prompt)
print(result.content)
