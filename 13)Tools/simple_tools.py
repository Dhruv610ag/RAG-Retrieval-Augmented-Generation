from langchain_core.tools import tool

@tool
def multiple(a:int,b:int)->int:
    """Multiple two numbers"""
    return a*b

result=multiple.invoke({"a":2,"b":5})
print(result)
print(multiple.name)
print(multiple.description)
print(multiple.args)
print(multiple.args_schema.model_json_schema())