from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence,RunnableParallel
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
import json
load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.7)

prompt1=PromptTemplate(
    template="Generate the tweet about the topic:{topic} and in the following format:\n'Tweet: <tweet_content>'",
    input_variables=["topic"]
)

prompt2=PromptTemplate(
    template="generate a linkedin post about the {topic} and in the following format:\n'Linkedin Post: <post_content>'",
    input_variables=["topic"]
)
parser=StrOutputParser()

chain=RunnableParallel({
    "tweet":RunnableSequence(prompt1,model,parser),
    "linkedin_post":RunnableSequence(prompt2,model,parser)
})

result=chain.invoke({"topic":"Runnable Parrallel in Langchain"})

print(json.dumps(result,indent=4))