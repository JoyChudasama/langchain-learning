from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence

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
    
    format_prompt = RunnableLambda(lambda x: prompt.format_prompt(**x))
    invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
    parse_output = RunnableLambda(lambda x: x.content)

    chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

    return chain.invoke({"topic": "ai", "joke_count": 2})

if __name__ == "__main__":
    res = main()
    print(res, '\n')