import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

def get_db()->Chroma:   
    current_dir = os.path.dirname(os.path.abspath(__file__))
    books_dir = os.path.join(current_dir, "books")
    db_path = os.path.join(current_dir, "db")
    persistent_directory = os.path.join(db_path, "chroma_db_with_metadata")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))

    if os.path.exists(persistent_directory):
        return Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
        
    books = [b for b in os.listdir(books_dir) if b.endswith(".txt")]
    documents = []

    for book in books:
        book_dir = os.path.join(books_dir, book)
        loader = TextLoader(book_dir, encoding='utf-8')
        book_docs = loader.load()
        
        for doc in book_docs:
            doc.metadata = {"source": book}
            documents.append(doc)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=300)
    docs = text_splitter.split_documents(documents)

    return Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)

def get_retriever(db:Chroma, search_type:str, search_kwags:dict)->list:
    return db.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwags,
    )
    
def get_model(model_name="mistral:latest") -> ChatOllama:
    return ChatOllama(
        model=model_name,
        temperature=0.1,
    )

def get_contextualize_q_prompt()->ChatPromptTemplate:
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question which might reference context in the chat history, "
        "formulate a standalone question which can be understood without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    return ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

def get_qa_prompt()->ChatPromptTemplate:
    qa_system_prompt = (
        "You are a geat assistant for question-answering tasks. "
        "You have Masters degree in finding answer of a given question from given context. "
        "Use the following pieces of retrieved context to answer the "
        "question. If you don't know the answer, just say that you don't know. "
        "Use three sentences maximum and keep the answer concise."
        "\n\nContext: \n{context}"
    )

    return ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

def get_rag_chain()->create_retrieval_chain:
    model = get_model()
    db = get_db()
    retriever = get_retriever(db, 'similarity_score_threshold', {'k':3, 'score_threshold':0.35})
    qa_prompt = get_qa_prompt()
    contextualize_q_prompt = get_contextualize_q_prompt()

    history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)

    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

def main():
    chat_history = []
    current_dir = os.path.dirname(os.path.abspath(__file__))
    books_dir = os.path.join(current_dir, "books")
    init_message = "\n\nPDF BOT: Available books are as given below." + "".join([f"\n- {b.replace('.txt', '')}" for b in os.listdir(books_dir) if b.endswith(".txt")]) + "\n\nYou can ask questions regarding any of these books."
    print(init_message)

    while True:
        user_query = input('\nQuery: ').strip()
        
        if not user_query or user_query.lower() == 'qq':
            print("\n--- Exiting ---")
            break
        
        print("\n--- Thinking ---")
        res = get_rag_chain().invoke({"input": user_query, "chat_history": chat_history})
        
        print(f"PDF BOT: {res['answer']} \n")
        
        chat_history.append(HumanMessage(content=user_query))
        chat_history.append(SystemMessage(content=res["answer"]))

if __name__ == "__main__":
    main()