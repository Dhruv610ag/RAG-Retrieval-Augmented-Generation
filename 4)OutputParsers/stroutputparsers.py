from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.7)
"""
PromptTemplate- It is used to create and manage text-based prompts for LLMs by defining a template with placeholders that can be dynamically filled with specific values at runtime.
is mainly used for simple single-turn interactions with LLMs where a specific prompt needs to be generated based on variable inputs.
"""
#1st prompt template
template1=PromptTemplate(template="Write a detailed report on {topic}",
                         input_variables=["topic"])

prompt1=template1.invoke({
    "topic":"English premier Leaque 2023/2024"
})

result1=model.invoke(prompt1).content

print("Report Generated:\n",result1)
#2nd prompt template
template2=PromptTemplate(template="Write a 4 point summary on the following text: {text}",
                         input_variables=["text"])
prompt2=template2.invoke({
    "text":str(result1)
})

result2=model.invoke(prompt2).content
print("Summary:\n",result2)