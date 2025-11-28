"""
What is an LLM chain?
An LLM chain is a sequence of operations involving a large language model (LLM).
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# MODEL
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# PROMPT
prompt1 = PromptTemplate(
    template="Suggest a catchy blog title about the following topic: {topic}",
    input_variables=["topic"]
)

# NEW LLMChain (LCEL way)
chain = prompt1 | model | StrOutputParser()

# RUN
topic = "3I Atlas Interstellar Object"
response = chain.invoke({"topic": topic})

print(response)
