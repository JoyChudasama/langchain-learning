from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage


# template_multiple = """You are a helpful assistant. Human: Tell me a {adjective} story about a {animal} Assistant:"""
# prompt_multiple = ChatPromptTemplate.from_template(template_multiple)
# prompt = prompt_multiple.invoke({"adjective": "funny", "animal": "panda"})
# print(prompt)


messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes."),
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "ai", "joke_count": 2})
print(prompt)