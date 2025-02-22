from langchain_ollama.chat_models import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage


@tool
def multiply(a: int, b: int) -> str:
    """Multiply two numbers given as strings."""
    return str(a * b)
   
def get_model(model_name="mistral:latest") -> ChatOllama:
    return ChatOllama(
        model=model_name,
        temperature=0.8,
        num_predict=256,
        format="json",
    )

def main(user_query: list) -> str: 
    ollama = get_model().bind_tools([multiply])
    response = ollama.invoke(user_query)
    return response.content if response else "No response received."


if __name__ == "__main__":
    # user_query_str = input('Query: ')
    user_query = [HumanMessage(content='hello world')]
    res = main(user_query)
    print(res)