from dotenv import load_dotenv
from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate

load_dotenv()

def get_model(model_name="mistral:latest") -> ChatOllama:
    return ChatOllama(
        model=model_name,
        temperature=1,
    )

def get_prompt() -> ChatPromptTemplate:
    messages = [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
    prompt_template = ChatPromptTemplate.from_messages(messages)
    return prompt_template.invoke({"topic": "ai", "joke_count": 2})

def main(messages: list) -> str:  
    return get_model().invoke(messages).content

if __name__ == "__main__":
    prompt = get_prompt()
    res = main(prompt)
    print(res, '\n')