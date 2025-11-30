from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
model=ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
prompt1=PromptTemplate(
    template="Write a joke about the topic:{topic}",
    input_variables=["topic"]
)

prompt2=PromptTemplate(
    template="Explain the following joke in about 100-120 words:\n{joke}",
    input_variables=["joke"]
)

parser=StrOutputParser()

chain=RunnableSequence(
    prompt1,model,parser,prompt2,model,parser
)

result=chain.invoke({"topic":"Technology"})\
    
print("Final Explanation:\n",result)
