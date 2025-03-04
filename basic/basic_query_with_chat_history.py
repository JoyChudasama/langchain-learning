from dotenv import load_dotenv
from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

def get_model(model_name="mistral:latest") -> ChatOllama:
    return ChatOllama(
        model=model_name,
        temperature=0.5,
    )

def main(messages: list) -> str:  
    return get_model().invoke(messages).content

if __name__ == "__main__":
    messages=[
        SystemMessage(content='You are a helpful assistant that does what human asks for to achieve their day to day life goals'),
    ]

    while True:
        user_query = input('\nQuery: ').strip()
        if not user_query or user_query.lower() == 'qq':
            print("Exiting...")
            break
        print("Thinking...")
        messages.append(HumanMessage(content=user_query))
        res = main(messages=messages)
        messages.append(AIMessage(res))
        print(res, '\n')