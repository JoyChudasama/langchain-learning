from langchain_ollama import ChatOllama


def get_model(model_name="mistral"):
    return ChatOllama(
        model=model_name,
        temperature=0.8,
        num_predict=256,
        format="json",
    )

def chat(messages): 
    llm = get_model("mistral")
    return llm.invoke(messages).content


def main(): 
    messages = [
        ("human", "Return a query for the weather in a random location and time of day with two keys: location and time_of_day. Respond using JSON only."),
    ]
    res =  chat(messages)
    print(res)


if __name__ == "__main__":
    main()
