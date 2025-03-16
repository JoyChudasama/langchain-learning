import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

def init_db(books_dir:str, persistent_directory:str)->Chroma:   

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))

    if os.path.exists(persistent_directory):
        print("\n--- Loading existing ChromaDB ---")
        return Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
        
    print("\n--- Processing and embedding documents ---")
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

def get_releavent_docs(query:str, db:Chroma, search_type:str, search_kwags:dict)->list:
    print("\n--- Searching RAG ---")
    retriever = db.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwags,
    )
    
    return retriever.invoke(query)
    
def get_model(model_name="mistral:latest") -> ChatOllama:
    return ChatOllama(
        model=model_name,
        temperature=0.5,
    )

def main(user_query:str)->list:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    books_dir = os.path.join(current_dir, "books")
    db_path = os.path.join(current_dir, "db")
    persistent_directory = os.path.join(db_path, "chroma_db_with_metadata")
    db = init_db(books_dir, persistent_directory)
    docs = get_releavent_docs(user_query, db, 'similarity_score_threshold', {'k':3, 'score_threshold':0.35})
    model = get_model()

    sys_prompt='You are a helpful assistant who has Masters Degree in finding answers from given Relevant Documents.'
    user_prompt = (
        "\n\nCheckout these documents which will help you answer question." 
        + f"\n\nQuestion: {user_query}"
        + "\n\nRelevant Documents:\n"
        + "\n\n".join([f"\nDocument ID: {i+1}\nDocument Source: {doc.metadata.get('source', 'Unknown')}\n\n{doc.page_content}\n" for i, doc in enumerate(docs) if doc])
        + "\n\nPlease provide an answer based only on the provided documents. In your answer include Document ID and Document Source if available on the first line and second line should be the actual answer. Also your answer should be short and concise with bits of rephrased question in it unless specified otherwise. If the answer is not found in the releavent documents, respond with 'I'm not sure'."
    )
    
    return model.invoke([SystemMessage(sys_prompt), HumanMessage(user_prompt)]).content

if __name__ == "__main__":
 
    while True:
        user_query = input('\nQuery: ').strip()
        
        if not user_query or user_query.lower() == 'qq':
            print("\n--- Exiting ---")
            break

        print("\n--- Thinking ---")
        res = main(user_query)
        print(res, '\n')

