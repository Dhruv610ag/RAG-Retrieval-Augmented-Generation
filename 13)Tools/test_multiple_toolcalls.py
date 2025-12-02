from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.tools import InjectedToolArg
from typing import Annotated

load_dotenv()

@tool
def get_conversion_rate(base_currency: str, target_currency: str) -> float:
    """Return the real time currency conversion rate from base currency to the target currency."""
    conversion_rate = 140.3   
    return conversion_rate

@tool
def convert(base_currency_value: float, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
    """Convert the base currency value to the target currency value."""
    return base_currency_value * conversion_rate

# LLM with tools bound
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
llm_with_tools = llm.bind_tools([get_conversion_rate, convert])

query = "what is the conversion factor b/w USD and NPR and based on it, can you convert 10 USD to NPR?"

messages = [HumanMessage(content=query)]

ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)

conversion_rate = None  # track rate for second tool call

while ai_msg.content == "" and ai_msg.tool_calls:
    tool_messages = []
    
    for tool_call in ai_msg.tool_calls:
        tool_name = tool_call['name']
        tool_id = tool_call['id']
        tool_arg = tool_call['args']

        print(f"Executing tool: {tool_name} with args: {tool_arg}")

        if tool_name == "get_conversion_rate":
            result = get_conversion_rate.invoke(tool_arg)
            conversion_rate = result
            tool_output = str(result)

        elif tool_name == "convert":
            # inject conversion_rate if missing
            if "conversion_rate" not in tool_arg:
                if conversion_rate is None:
                    raise ValueError("Tool 'convert' requires conversion_rate but none found.")
                tool_arg["conversion_rate"] = conversion_rate

            result = convert.invoke(tool_arg)
            tool_output = str(result)

        else:
            tool_output = f"Unknown tool: {tool_name}"

        tool_msg = ToolMessage(content=tool_output, tool_call_id=tool_id)
        tool_messages.append(tool_msg)

    # append each tool message individually
    messages.extend(tool_messages)

    # call LLM again
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)

result = messages[-1]
print("\nFinal Answer:\n", result.content)
