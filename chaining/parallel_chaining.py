from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda

def get_model(model_name="mistral:latest") -> ChatOllama:
    return ChatOllama(
        model=model_name,
        temperature=0.75,
    )

def init_prompt_template(product_name:str) -> ChatPromptTemplate:
    messages = [
        ("system", "You are an expert product reviewer."),
        ("human", "Give description of the {product_name} and list the main features of the product {product_name}."),
    ]

    return ChatPromptTemplate.from_messages(messages)

def describe_product(product_detail)->ChatPromptTemplate:
    messages = [
        ("system", "You are an expert in sumarrizing description and list of pros and cons in a short 2 to 3 sentence paragraph. You use your own language to construct the summary. You also rephrase the sentences. Make it fun to read. Use emojis unless specified not to."),
        ("human", "Given the product description, pros and cons of the product: {product_detail}, summarize it."),
    ]

    return ChatPromptTemplate.from_messages(messages)

def analyze_pros(product_detail:str)->ChatPromptTemplate:
    messages = [
        ("system", "You are an expert product reviewer. Who finds pros from product features. By default, list 5 pros unless a specific number given."),
        ("human", "Given these features: {product_detail}, list 5 pros of these features unless."),
    ]

    return ChatPromptTemplate.from_messages(messages)

def analyze_cons(product_detail:str)->ChatPromptTemplate:
    messages = [
        ("system", "You are an expert product reviewer. Who finds cons from product features. By default, list 5 cons unless a specific number given."),
        ("human", "Given these features: {product_detail}, list the cons of these features."),
    ]

    return ChatPromptTemplate.from_messages(messages)

def combine_results(description: ChatPromptTemplate, pros: ChatPromptTemplate, cons: ChatPromptTemplate)->str:
    return f"Description:\n{description}\n\nPros:\n{pros}\n\nCons:\n{cons}"

    
def main(user_query:str) -> str:
    prompt_template = init_prompt_template(user_query)
    model = get_model()
    description_chain = RunnableLambda(lambda x: describe_product(user_query) | model | StrOutputParser())
    pros_branch_chain = RunnableLambda(lambda x: analyze_pros(x) | model | StrOutputParser())
    cons_branch_chain = RunnableLambda(lambda x: analyze_cons(x) | model | StrOutputParser())

    chain = (
        prompt_template
        | model
        | StrOutputParser()
        | RunnableParallel(branches={"description":description_chain, "pros": pros_branch_chain, "cons": cons_branch_chain})
        | RunnableLambda(lambda x: combine_results(x["branches"]["description"], x["branches"]["pros"], x["branches"]["cons"]))
    )

    return chain.invoke({"product_name":user_query})

if __name__ == "__main__":

    while True:
        user_query = input('\nQuery: ').strip()
        if not user_query or user_query.lower() == 'qq':
            print("Exiting...")
            break
        print("Thinking...")
        res = main(user_query)
        print(res, '\n')

   