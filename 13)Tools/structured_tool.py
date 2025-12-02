from langchain_core.tools import StructuredTool
from pydantic import BaseModel,Field

class MultiplyInput(BaseModel):
    a:int=Field(description="The frst number to multiple")
    b:int=Field(description="the second number to multiple")
    
def multiple_func(a:int,b:int)->int:
    return a*b

multiply_tool=StructuredTool(
    func=multiple_func,
    name="multiple",
    description="Multiple two numbers",
    args_schema=MultiplyInput
)

result=multiply_tool.invoke(
    {
        "a":2,
        "b":3
    }
)
print(result)
