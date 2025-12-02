from langchain_core.tools import tool

@tool
def add(a:int,b:int):
    """Addition of two numbers"""
    return a+b

@tool
def mul(a:int,b:int):
    """Multiplication of two numbers"""
    return a*b

@tool
def sub(a:int,b:int):
    """Subtract of the Two numbers"""
    return a-b    
class Mathtoolkit:
    def get_tools(self):
        return [add,mul,sub]

toolkit=Mathtoolkit()
tools=toolkit.get_tools()
for tl in tools:
    print(tl.name,"--->",tl.description)