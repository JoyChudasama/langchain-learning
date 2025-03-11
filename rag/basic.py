import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

def init_db(file_path:str, persistent_directory:str,)->Chroma:   

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    if os.path.exists(persistent_directory):
        print("\n--- Loading existing ChromaDB ---")
        db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
    else:
        print("\n--- Processing and embedding documents ---")
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)

        db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
        db.persist() 
    
    return db

def releavent_docs(query:str, db:Chroma)->list:
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.4},
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
    file_path = os.path.join(current_dir, "books", "book_of_baseball.txt")
    persistent_directory = os.path.join(current_dir, "db", "chroma_db")
    db = init_db(file_path, persistent_directory)

    while True:
        user_query = input('\nQuery: ').strip()
        if not user_query or user_query.lower() == 'qq':
            print("\n--- Exiting ---")
            break
        print("\n--- Thinking ---")
        res = main(user_query, db)
        print(res, '\n')
