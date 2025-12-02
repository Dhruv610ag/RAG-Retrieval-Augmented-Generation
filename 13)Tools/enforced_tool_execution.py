from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()
model=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

@tool
def mul(a:int,b:int):
    """Multiplication of two numbers"""
    return a*b

llm_with_tools=model.bind_tools([mul])

user_query="Can you multiple 10 and 20"
query=HumanMessage(user_query)

messages=[query]

result=llm_with_tools.invoke(messages)
messages.append(result)
tool_result=mul.invoke(result.tool_calls[0])

messages.append(tool_result)
final_result=llm_with_tools.invoke(messages)
print(final_result.content)
