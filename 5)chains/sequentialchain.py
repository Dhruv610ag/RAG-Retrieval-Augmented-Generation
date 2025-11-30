from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

promt1=PromptTemplate(
    template="Generate 3 detailed report on the {topic}.",
    input_variables=["topic"])
promt2=PromptTemplate(
    template="generate 3 point summary on following text: {text}",
    input_variables=["text"]
)
parser=StrOutputParser()
#sequential chain invokation
chain=promt1|model|promt2|model|parser
response=chain.invoke({"topic":"Artificial Intelligence"})
print(response)

"""
these seuential chain will first generate a detailed report on the given topic using the first prompt and the language model.
they basically work line by line like the first prompt first and then the second prompt
whereas conditional chain will decide which prompt to use based on the output of the previous step.
they are more robust and better then the sequential chain and clser to human like decision making 
"""