from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
load_dotenv()
from pydantic import BaseModel,Field
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
# --- Step 1: Define Pydantic Model ---

"""
Pydantic output parser are the upgrade version of the structured output parsers.
They leverage Pydantic models to define the expected structure of the output from LLMs. 
They ensure that the output adheres to the defined schema, providing type validation and data integrity.
Pydantic is used for the data validation and also check the types of fields automatically when the output is parsed.
"""
class Person(BaseModel):
    name: str = Field(description="Name of the Person")
    age: int = Field(gt=18,lt=100,description="Age of the Person")
    city: str = Field(description="City where the person lives")
# --- Step 2: Create Pydantic Output Parser ---
parser = PydanticOutputParser(pydantic_object=Person)
template = ChatPromptTemplate.from_template(
    """
Give me the name, age, and city of a fictional person from {place}.
Age must be >18 and <100.
Return ONLY valid JSON in this format:
{response_format}
"""
)
template = template.partial(response_format=parser.get_format_instructions())
# --- Step 3: Build the chain
chain = template | model | parser
result = chain.invoke({
    "place": "nepal"
})  
print("Final information about the person is:\n", result)
"""
The main difference between the structured output parser and the Pydantic output parser lies in the way they define and enforce the output schema.
In the structured output parser, the schema is defined using ResponseSchema objects, which specify the name and description of each field. The parser then ensures that the output from the LLM adheres to this schema.
In contrast, the Pydantic output parser uses a Pydantic model to define the schema. This allows for more complex data structures and automatic type validation, as Pydantic models can include various field types, nested models, and validation logic.    
Pydantic addes an additional layer of validation and type checking, making it a more robust choice for scenarios where data integrity is crucial.
"""