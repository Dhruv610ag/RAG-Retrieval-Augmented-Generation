from langchain_google_genai import ChatGoogleGenerativeAI
from  dotenv import load_dotenv
from typing import TypedDict

load_dotenv()
model=ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

class Review(TypedDict):
    summary:str
    sentiment:str

structured_model=model.with_structured_output(Review)
prompt="The product was excellent and met all my expectations."

response=structured_model.invoke(prompt)
print(response)
