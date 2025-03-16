import os
from dotenv import load_dotenv

from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    TextSplitter,
    TokenTextSplitter,
)
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
# from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

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

    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_splitter = SentenceTransformersTokenTextSplitter(chunk_size=1000)
    docs = text_splitter.split_documents(documents)

    return Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)

def releavent_docs(query:str, db:Chroma)->list:
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.1},
    )

    return retriever.invoke(query)

def main(user_query:str, db:Chroma)->list:
    docs = releavent_docs(user_query, db)

    print("\n--- Relevant Documents ---")
    for i, doc in enumerate(docs, 1):
        print(f"Document {i}:\n{doc.page_content}\n")
        if doc.metadata:
            print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    books_dir = os.path.join(current_dir, "books")
    db_path = os.path.join(current_dir, "db")
    persistent_directory = os.path.join(db_path, "chroma_db_with_metadata")
    db = init_db(books_dir, persistent_directory)

    while True:
        user_query = input('\nQuery: ').strip()
        if not user_query or user_query.lower() == 'qq':
            print("\n--- Exiting ---")
            break
        print("\n--- Thinking ---")
        res = main(user_query, db)
        print(res, '\n')
