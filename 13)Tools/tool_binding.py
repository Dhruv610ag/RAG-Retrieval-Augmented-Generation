from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
model=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def mul(a:int,b:int):
    """Multiplication of two numbers"""
    return a*b

llm_with_tools=model.bind_tools([mul])

result=llm_with_tools.invoke("Can you multiple 4 with 16 ")
print(result)
