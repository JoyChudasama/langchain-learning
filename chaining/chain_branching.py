from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda

def get_model(model_name="mistral:latest") -> ChatOllama:
    return ChatOllama(
        model=model_name,
        temperature=1,
    )

def positive_feedback_prompt_template(feedback:str)->ChatPromptTemplate:
    messages = [
        ("system", "You are a helpful assistant. You are great at writing reply for positive feedback in very human like friendly language."),
        ("human", "Generate a thank you note for this positive feedback: {feedback}."),
    ]

    return ChatPromptTemplate.from_messages(messages)

def negative_feedback_prompt_template(feedback:str)->ChatPromptTemplate:
    messages = [
        ("system", "You are a helpful assistant. You are great at writing reply for negative feedback in very human like friendly language."),
        ("human", "Generate a response addressing this negative feedback: {feedback}."),
    ]

    return ChatPromptTemplate.from_messages(messages)

def neutral_feedback_prompt_template(feedback:str)->ChatPromptTemplate:
    messages = [
        ("system", "You are a helpful assistant. You are great at writing reply for neutral feedback in very human like friendly language."),
        ("human", "Generate a request for more details for this neutral feedback: {feedback}."),
    ]

    return ChatPromptTemplate.from_messages(messages)

def escalate_feedback_prompt_template(feedback:str)->ChatPromptTemplate:
    messages = [
        ("system", "You are a helpful assistant."),
        ("human", "Generate a message to escalate this feedback to a human agent: {feedback}."),
    ]

    return ChatPromptTemplate.from_messages(messages)

def classification_prompt_template(feedback:str)->ChatPromptTemplate:
    messages = [
        ("system", "You are a helpful assistant."),
        ("human", "Classify the sentiment of this feedback as positive, negative, neutral, or escalate: {feedback}."),
    ]

    return ChatPromptTemplate.from_messages(messages)

def get_branches(model:ChatOllama)->RunnableBranch:

    return RunnableBranch(
        (
            lambda x: "positive" in x,
            positive_feedback_prompt_template | model | StrOutputParser()  
        ),
        (
            lambda x: "negative" in x,
            negative_feedback_prompt_template | model | StrOutputParser()  
        ),
        (
            lambda x: "neutral" in x,
            neutral_feedback_prompt_template | model | StrOutputParser() 
        ),
        escalate_feedback_prompt_template | model | StrOutputParser()
    )
    
def main(user_query:str) -> str:
    model = get_model()
    branches = get_branches(model)

    classification_chain = classification_prompt_template | model | StrOutputParser()
    chain = classification_chain | branches
    
    return chain.invoke({"feedback":user_query})

if __name__ == "__main__":

    while True:
        user_query = input('\nQuery: ').strip()
        if not user_query or user_query.lower() == 'qq':
            print("Exiting...")
            break
        print("Thinking...")
        res = main(user_query)
        print(res, '\n')

   