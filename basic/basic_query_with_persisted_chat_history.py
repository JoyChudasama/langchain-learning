from dotenv import load_dotenv
import os
from langchain_ollama.chat_models import ChatOllama
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory

load_dotenv()

def get_model(model_name="mistral:latest") -> ChatOllama:
    return ChatOllama(
        model=model_name,
        temperature=0.5,
    )

def main(messages: list) -> str:  
    return get_model().invoke(messages).content

if __name__ == "__main__":

    username = input('\nUsername: ').strip()

    PROJECT_ID = os.environ.get("PROJECT_ID")
    SESSION_ID = username
    COLLECTION_NAME = 'chat_history'

    client = firestore.Client(project=PROJECT_ID)
    chat_history = FirestoreChatMessageHistory(
        session_id=SESSION_ID,
        collection=COLLECTION_NAME,
        client=client,
    )
    
    sys_message='You are a helpful assistant that does what human asks for to achieve their day to day life goals'
    chat_history.add_user_message(sys_message)

    while True:
        user_query = input('\nQuery: ').strip()
        if not user_query or user_query.lower() == 'qq':
            print("Exiting...")
            break
        print("Thinking...")
        chat_history.add_user_message(user_query)
        res = main(messages=chat_history.messages)
        chat_history.add_ai_message(res)

        print(res, '\n')