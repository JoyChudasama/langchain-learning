from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

def get_model(model_name="mistral:latest") -> ChatOllama:
    return ChatOllama(
        model=model_name,
        temperature=0.75,
    )

def get_prompt() -> ChatPromptTemplate:
    messages = [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
    
    return ChatPromptTemplate.from_messages(messages)

def main() -> str:
    prompt = get_prompt()
    model = get_model()
    chain = prompt | model | StrOutputParser()

    return chain.invoke({"topic": "ai", "joke_count": 3})

if __name__ == "__main__":
    res = main()
    print(res, '\n')