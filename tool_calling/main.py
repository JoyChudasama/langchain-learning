from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
from tool_calling import call_tools

def get_model(model_name="mistral:latest") -> ChatOllama:
    return ChatOllama(
        model=model_name,
        temperature=0.5,
    )

def main(user_query: str) -> str:  
    return call_tools(get_model(), user_query)  

if __name__ == "__main__":

    while True:
        user_query = input('\nQuery: ').strip()
        if not user_query or user_query.lower() == 'qq':
            print("Exiting...")
            break
        print("Thinking...")
        res = main(HumanMessage(content=user_query))
        print(res, '\n')