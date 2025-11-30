from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.7)
"""
PromptTemplate- It is used to create and manage text-based prompts for LLMs by defining a template with placeholders that can be dynamically filled with specific values at runtime.
is mainly used for simple single-turn interactions with LLMs where a specific prompt needs to be generated based on variable inputs.
"""
#1st prompt template
template1=PromptTemplate(template="Write a detailed report on {topic}",
                         input_variables=["topic"])


#2nd prompt template
template2=PromptTemplate(template="Write a 4 point summary on the following text: {text}",
                         input_variables=["text"])

parser=StrOutputParser()
chain=template1 | model | template2 | model | parser

result=chain.invoke({
    "topic":"English premier Leaque 2023/2024"
})
print("Final Summary:\n",result)