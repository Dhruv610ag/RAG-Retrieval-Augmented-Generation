from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

"""
Chat prompt template is used for the structured,multi-turn it is defined for the chatbots and the other interactive application by defining a sequence of messages from different roles such as 
system, user and assistant
"""
load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful {domain} assistant."),
    ("human", "Explain me in simple terms the concept of {topic}.")
])
prompt = chat_template.invoke({
    "domain": "physics",
    "topic": "quantum entanglement"
})
result = model.invoke(prompt)
print(result.content)
