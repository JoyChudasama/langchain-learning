from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import ToolMessage
from calculation_tools import multiply, add, calculate_based_on_income, calculate_equally

def call_tools(model:ChatOllama, user_query: str) -> str:  
    msgs = [user_query]
    ollama = model.bind_tools(tools=[multiply, add, calculate_based_on_income, calculate_equally])
    ai_msg = ollama.invoke(input=msgs)
    msgs.append(ai_msg)

    print("DEBUG: ", ai_msg.tool_calls)

    for tool_call in ai_msg.tool_calls:
        selected_tool = {"add": add, "multiply": multiply, "calculate_based_on_income":calculate_based_on_income, "calculate_equally": calculate_equally}[tool_call["name"].lower()]
        tool_output = selected_tool.invoke(tool_call["args"])
        msgs.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))

    response = ollama.invoke(msgs)

    return response.content if response else "No response received."  