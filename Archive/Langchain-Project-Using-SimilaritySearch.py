import os
import glob
import re
import datetime
import uuid
import shutil
from pymongo import MongoClient, errors
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_core.documents import Document


#Config MongoDB

MONGO_URL="mongodb+srv://surajyaligar2004_db_user:LsMU0ZqDZJuyG0Si@documents.pji2q4k.mongodb.net/?appName=Documents"
DB_NAME="rag_notebook_db"
COLLECTION_NAME="Documents"

VECTOR_STORE_ROOT="./vector_stores"

try:
    client = MongoClient(MONGO_URL,serverSelectionTimeoutMS=5000)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    client.server_info()
    print("Connected to MongoDB successfully.")
except errors.ServerSelectionTimeoutError:
    print("Failed to connect to MongoDB.")
    exit()

if not os.path.exists(VECTOR_STORE_ROOT):
    os.makedirs(VECTOR_STORE_ROOT)

#Functions


def count_tokens(text):
    try:
        if not text:return 0
        return len(text)
    except Exception:
        return 0
    
def get_embeddings():
    try:
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        print(f"Error getting embeddings: {e}")
        raise e
    
def beautify_text(text):
    text = re.sub(r'([a-zA-Z])([A-Z])', r'\1 \2', text) # Fix "ThisIs"
    text = re.sub(r'\s{2,}', ' ', text) # Normalize spaces
    text = text.replace("●", "\n• ") # Fix bullets
    return text.strip()
    
#upload function

def load_single_file(file_path):
    try:
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path,encoding='utf-8')
        elif file_path.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            print(f"Unsupported file type: {file_path}")
            return None
        
        return loader.load()
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def process_and_store_document(folder_path,embeddings):
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return
    
    files=[]
    for ext in ('*.pdf', '*.txt', '*.docx'):
        files.extend(glob.glob(os.path.join(folder_path, ext)))

    print(f"Found {len(files)} files in {folder_path}.")

    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)

    for file_path in files:
        filename=os.path.basename(file_path)
#function
        if collection.find_one({"filename":filename}):
            print(f"Document {filename} already exists in the database.")
            continue

        print("Processing file:", filename)

        try:
            docs=load_single_file(file_path)
            if not docs:
                continue
            
            chunks=text_splitter.split_documents(docs)

            full_text=" ".join([d.page_content for d in chunks])
            tokens=count_tokens(full_text)

            vector_store=FAISS.from_documents(chunks,embeddings)

            unique_id=str(uuid.uuid4())
            save_path=os.path.join(VECTOR_STORE_ROOT,unique_id)
            vector_store.save_local(save_path)
            current_time = datetime.datetime.now(datetime.timezone.utc).isoformat()

            record={
                "_id":unique_id,
                "filename":filename,
                "file_path":os.path.abspath(file_path),
                "vector_path":os.path.abspath(save_path),
                "token_count":tokens,
                "created_at":current_time
            }
            collection.insert_one(record)
            print(f"Successfully processed and stored document: {filename}")
        
        except Exception as e:
            print(f"Error processing file {filename}: {e}")

#Query function

def list_documents():
    cursor = collection.find({}, {"_id": 1, "filename": 1, "created_at": 1, "token_count": 1})
    print(f"\n{'ID':<38} | {'Filename':<20} | {'Tokens':<10} | {'Created'}")
    print("-" * 85)

    docs_found = False
    for doc in cursor:
        docs_found = True
        print(f"{doc['_id']:<38} | {doc['filename'][:20]:<20} | "
              f"{doc.get('token_count',0):<10} | {doc['created_at'][:10]}")

    if not docs_found:
        print("No documents found.")


def load_vector_store(record, embeddings):
    try:
        return FAISS.load_local(
            record['vector_path'],
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"Critical Error loading vector store: {e}")
        return None


def chat_loop(record, vector_store):
    print(f"\n💬  Chatting with {record['filename']} (Type 'exit' to stop)\n")

    while True:
        query = input(" Question: ").strip()
        if query.lower() == 'exit':
            break

        results = vector_store.similarity_search_with_score(query, k=3)
        print("\n Top Matches:")
        
        for i, (doc, score) in enumerate(results, start=1):
            page_num = doc.metadata.get("page", "N/A") 
            if isinstance(page_num, int): 
                page_num += 1 
            
            print(f"\n--- Match {i} (Score: {score:.4f}) | 📄 Page: {page_num} ---")
            
            print(beautify_text(doc.page_content))
        print("-" * 50)


def search_database_and_chat(embeddings):
    while True:
        print("\n" + "="*40)
        print("  DATABASE MENU ")
        print("="*40)
        print("1. List all Documents")
        print("2. Select Document by ID to Chat")
        print("3. Back to Main Menu")

        choice = input("\nEnter choice: ").strip()

        if choice == '1':
            list_documents()

        elif choice == '2':
            doc_id = input("Enter Document ID (UUID): ").strip()
            record = collection.find_one({"_id": doc_id})

            if not record:
                print("Document ID not found in database.")
                continue

            print(f"\nLoaded: {record['filename']}")
            print(f"Vector Path: {record['vector_path']}")

            vector_store = load_vector_store(record, embeddings)
            if vector_store:
                chat_loop(record, vector_store)

        elif choice == '3':
            break

        else:
            print("Invalid choice. Try again.")



def main():
    print("Initializing Embedding Model (this may take a moment)...")
    try:
        embeddings = get_embeddings()
    except Exception:
        return

    while True:
        print("\n" + "="*50)
        print("MongoDB+FAISS Document Chat Application")
        print("="*50)
        print("1. New files Folder (Process new files)")
        print("2. Query Database (Chat with doc)")
        print("3. Quit")

        action = input("\nSelect Action: ")

        if action == '1':
            folder = input("Enter folder path containing docs: ").strip()
            process_and_store_document(folder, embeddings)
        elif action == '2':
            search_database_and_chat(embeddings)
        elif action == '3':
            print("Goodbye!")
            break
        else:
            print("Invalid selection.")

if __name__ == "__main__":
    main()