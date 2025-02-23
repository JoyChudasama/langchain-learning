from langchain_ollama.chat_models import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers."""
    return str(a * b)

@tool
def add(a: int, b: int) -> int:
    """Adds two numbers."""
    return str(a + b)

def get_model(model_name="mistral:latest") -> ChatOllama:
    return ChatOllama(
        model=model_name,
        temperature=0.5,
    )

def main(user_query: str) -> str:  
    msgs = [user_query]
    ollama = get_model().bind_tools(tools=[multiply, add])
    ai_msg = ollama.invoke(input=msgs)
    msgs.append(ai_msg)
    print(ai_msg.tool_calls)

    for tool_call in ai_msg.tool_calls:
        selected_tool = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
        tool_output = selected_tool.invoke(tool_call["args"])
        msgs.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))

    response = ollama.invoke(msgs)

    return response.content if response else "No response received."  

if __name__ == "__main__":
    user_query=input('Query: ')
    res = main(HumanMessage(content=user_query.strip()))
    print(res)