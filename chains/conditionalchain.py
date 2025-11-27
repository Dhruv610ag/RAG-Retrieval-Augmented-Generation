from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
from langchain_core.output_parsers import PydanticOutputParser,StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableBranch,RunnableLambda
from typing import Literal
# conditional chaining makes our app or program more robust and intelligent by allowing it to make decisions based on the output of previous steps.
model=ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
parser=StrOutputParser()

class Feedback(BaseModel):
    sentiment:Literal['positive','negative','neutral']=Field(description="The sentiment of the feedback")

parser2=PydanticOutputParser(pydantic_object=Feedback)

prompt1=PromptTemplate(
    template="Analyze the sentiment of the following feedback as the positive,negative and neutral: {feedback} and provide the response in the following format {response}",
    input_variables=["feedback"],
    partial_variables={"response":parser2.get_format_instructions()}
)
classifier_chain=prompt1|model|parser2

prompt2=PromptTemplate(
    template="Write an appropiatre feedbackto this positive feedback : {feedback}",
    input_variables=["feedback"]
)
prompt3=PromptTemplate(
    template="Write an appropiatre feedbackto this negative feedback : {feedback}", 
    input_variables=["feedback"]
)

branch_chain=RunnableBranch(
    (lambda x:x.sentiment=="positive",prompt2|model|parser),
    (lambda x:x.sentiment=="negative",prompt3|model|parser),
    RunnableLambda(lambda x:"No valid sentiment found in my review or feedback.")
)

chain=classifier_chain|branch_chain
response=chain.invoke({"feedback":"The product quality is excellent and I am very satisfied with my purchase."})
print(response)

