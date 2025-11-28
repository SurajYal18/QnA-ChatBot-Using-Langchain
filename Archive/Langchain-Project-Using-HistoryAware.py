import os
import datetime
import uuid
import warnings
import logging

# --- SUPPRESS WARNINGS ---
# Suppress the LangSmith/Pydantic UUID warning
warnings.filterwarnings("ignore", message=".*LangSmith now uses UUID v7.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
logging.getLogger("langsmith").setLevel(logging.ERROR)

from dotenv import load_dotenv
from pymongo import MongoClient, errors

# --- LANGCHAIN CORE ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# --- NEW CHAIN ARCHITECTURE (LCEL) ---
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# --- MONGODB HISTORY ---
from langchain_mongodb import MongoDBChatMessageHistory

# --- VECTOR STORE & DOCS ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, DirectoryLoader, CSVLoader
)

# Configuration
def load_app_configuration():
    load_dotenv()
    config = {
        "mongoUrl": os.getenv("MONGO_URL"),
        "googleApiKey": os.getenv("GOOGLE_API_KEY"),
        "dbName": os.getenv("DB_NAME", "rag_folder_db"),
        "collectionName": os.getenv("COLLECTION_NAME", "dab"), 
        "chatCollectionName": "chat_histories", 
        "sessionMetaCollection": "session_metadata", 
        "sourceDirectory": "./static",        
        "vectorStoreRoot": "./vector_stores", 
        "chunkSize": 1000, 
        "chunkOverlap": 100
    }
    if not config["mongoUrl"] or not config["googleApiKey"]:
        print("Error: Missing API Keys in .env file.")
        exit()
    if not os.path.exists(config["vectorStoreRoot"]):
        os.makedirs(config["vectorStoreRoot"])
    return config

# Database Functions 
def connect_to_mongodb(config):
    try:
        client = MongoClient(config["mongoUrl"], serverSelectionTimeoutMS=5000)
        client.server_info()
        return client[config["dbName"]] 
    except errors.ServerSelectionTimeoutError:
        print("Failed to connect to MongoDB.")
        exit()

# --- SESSION MANAGEMENT ---
def save_session_metadata(db, session_id, folder_id, folder_name):
    """Links a Session ID to a Folder Name in MongoDB."""
    meta_collection = db["session_metadata"]
    meta_collection.update_one(
        {"session_id": session_id},
        {
            "$set": {
                "session_id": session_id,
                "folder_id": str(folder_id),
                "folder_name": folder_name,
                "last_active": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
        },
        upsert=True
    )

def fetch_all_sessions(db):
    meta_collection = db["session_metadata"]
    return meta_collection.find().sort("last_active", -1)

def fetch_session_details(db, session_id):
    meta_collection = db["session_metadata"]
    return meta_collection.find_one({"session_id": session_id})

# --- FOLDER FUNCTIONS ---
def check_folder_exists_in_db(db, folderName):
    return db["dab"].find_one({"folderName": folderName}) is not None

def insert_folder_record(db, folderName, folderPath, vectorPath, tokenCount):
    uniqueId = str(uuid.uuid4())
    record = {
        "_id": uniqueId,
        "folderName": folderName,
        "folderPath": os.path.abspath(folderPath),
        "vectorPath": os.path.abspath(vectorPath),
        "tokenCount": tokenCount,
        "createdAt": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }
    try:
        db["dab"].insert_one(record)
        print(f"Database record created for: {folderName}")
    except Exception as e:
        print(f"Error inserting record: {e}")

def fetch_all_folders(db):
    return db["dab"].find({}, {"_id": 1, "folderName": 1})

def fetch_folder_by_id(db, folderId):
    return db["dab"].find_one({"_id": folderId})

# Core Processing
def count_tokens(textInput):
    return len(textInput) // 4 if textInput else 0

def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_gemini_llm(apiKey):
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=apiKey, temperature=0.3)

# Document Loading & Processing
def load_documents_from_folder(folderPath):
    allDocuments = []
    loader_mapping = {
        "**/*.pdf":  (PyPDFLoader, {}),
        "**/*.txt":  (TextLoader, {'encoding': 'utf-8'}),
        "**/*.docx": (UnstructuredWordDocumentLoader, {}),
        "**/*.csv":  (CSVLoader, {'encoding': 'utf-8'}),
    }
    try:
        for glob_pattern, (loader_cls, loader_kwargs) in loader_mapping.items():
            loader = DirectoryLoader(folderPath, glob=glob_pattern, loader_cls=loader_cls, loader_kwargs=loader_kwargs, silent_errors=True)
            loaded_docs = loader.load()
            if loaded_docs: allDocuments.extend(loaded_docs)
        return allDocuments
    except Exception as e:
        print(f"Error loading files: {e}")
        return []

def split_documents(documents, config):
    textSplitter = RecursiveCharacterTextSplitter(chunk_size=config["chunkSize"], chunk_overlap=config["chunkOverlap"])
    return textSplitter.split_documents(documents)

def create_vector_store(docChunks, emeddingModel):
    return FAISS.from_documents(docChunks, emeddingModel)

def save_vector_store(vectorStore, config):
    uniqueFolderId = str(uuid.uuid4())
    savePath = os.path.join(config["vectorStoreRoot"], uniqueFolderId)
    vectorStore.save_local(savePath)
    return savePath

def process_static_directory(db, embeddingModel, config):
    sourceDir = config["sourceDirectory"]
    if not os.path.exists(sourceDir):
        print(f"Source dir '{sourceDir}' not found.")
        return
    subFolders = [f.path for f in os.scandir(sourceDir) if f.is_dir()]
    print(f"Found {len(subFolders)} folders. Processing...")
    
    for subFolder in subFolders:
        folderName = os.path.basename(os.path.normpath(subFolder))
        if check_folder_exists_in_db(db, folderName):
            print(f"Skipping '{folderName}' (Exists).")
            continue
        print(f"Processing: {folderName}...")
        documents = load_documents_from_folder(subFolder)
        if not documents: continue
        docChunks = split_documents(documents, config)
        vectorStore = create_vector_store(docChunks, embeddingModel)
        if vectorStore:
            vectorSavePath = save_vector_store(vectorStore, config)
            insert_folder_record(db, folderName, subFolder, vectorSavePath, count_tokens(" ".join([d.page_content for d in docChunks])))
    print("Batch processing complete.")

# --- CHAT FUNCTIONS ---

def load_vector_store_local(vectorPath, embeddingModel):
    try:
        return FAISS.load_local(vectorPath, embeddingModel, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None

# New Conversational Chain with History
def create_conversational_chain_modern(vectorStore, llm, config):
    
    retriever = vectorStore.as_retriever(search_kwargs={"k": 5})

    # 1. Contextualize Question Prompt
    # This prompt reformulates the question based on history
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # Creates a retriever that is "history aware"
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 2. Answer Question Prompt
    # This prompt generates the final answer
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If the answer is not in the context, check the Chat History."
        "If you still don't know the answer, say that you don't know."
        "\n\n"
        "{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"), # This passes history to the FINAL step
        ("human", "{input}"),
    ])
    
    # Standard QA chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # 3. Final RAG Chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    # 4. Wrap with Message History (MongoDB)
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        return MongoDBChatMessageHistory(
            session_id=session_id,
            connection_string=config["mongoUrl"],
            database_name=config["dbName"],
            collection_name=config["chatCollectionName"]
        )

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    return conversational_rag_chain

def display_response(result):
    print("\n✨ Gemini Answer:")
    print(result["answer"])
    
    # Modern chains allow looking at 'context' in the result
    if "context" in result:
        print("\n--- Sources Used ---")
        seen = set()
        for doc in result["context"]:
            src = os.path.basename(doc.metadata.get("source", "unknown"))
            if src not in seen:
                print(f"📄 {src}")
                seen.add(src)
        print("-" * 50)

def chat_loop(db, dbRecord, vectorStore, llm, config, session_id):
    save_session_metadata(db, session_id, dbRecord["_id"], dbRecord["folderName"])

    print(f"\n💬 Chat Started | Topic: {dbRecord['folderName']}")
    print(f"🔑 Session ID: {session_id}")
    print("Type 'exit' to stop.\n")
    
    # Use the new chain creator
    rag_chain = create_conversational_chain_modern(vectorStore, llm, config)

    while True:
        query = input(" \nQuestion (or 'exit'): ").strip()
        if query.lower() == 'exit': break
        print("Thinking...", end="\r")
        try:
            # New Invoke Syntax
            # Pass session_id in configurable
            result = rag_chain.invoke(
                {"input": query},
                config={"configurable": {"session_id": session_id}}
            )
            display_response(result)
            save_session_metadata(db, session_id, dbRecord["_id"], dbRecord["folderName"])
        except Exception as e:
            print(f"Error: {e}")

# --- MENU FUNCTIONS ---

def start_new_chat(db, embeddingModel, config):
    folders = list(fetch_all_folders(db))
    if not folders:
        print("No folders found.")
        return
    print(f"\n{'ID':<38} | {'Folder Name':<20}")
    print("-" * 60)
    for doc in folders:
        print(f"{str(doc['_id']):<38} | {doc['folderName'][:20]:<20}")
        
    choice = input("\nEnter Folder ID: ").strip()
    record = fetch_folder_by_id(db, choice)
    
    if record:
        vStore = load_vector_store_local(record['vectorPath'], embeddingModel)
        llm = get_gemini_llm(config["googleApiKey"])
        if vStore and llm:
            new_session_id = str(uuid.uuid4())
            chat_loop(db, record, vStore, llm, config, new_session_id)
    else:
        print("Invalid ID.")

def resume_previous_session(db, embeddingModel, config):
    sessions = list(fetch_all_sessions(db))
    if not sessions:
        print("\n No saved session history found yet. Start a new chat first!")
        return

    print(f"\n{'Session ID':<38} | {'Folder Name':<20} | {'Last Active'}")
    print("-" * 80)
    
    for s in sessions:
        s_id = s.get('session_id', 'Unknown')
        f_name = s.get('folder_name', 'Unknown')
        l_active = s.get('last_active', '')[:16].replace('T', ' ')
        print(f"{s_id:<38} | {f_name[:20]:<20} | {l_active}")

    s_choice = input("\nEnter Session ID to resume (or 'b' to back): ").strip()
    if s_choice.lower() == 'b': return

    session_meta = fetch_session_details(db, s_choice)
    if not session_meta:
        print("Session ID not found.")
        return

    folder_id = session_meta.get('folder_id')
    record = fetch_folder_by_id(db, folder_id)
    
    if not record:
        print("Error: The folder associated with this session no longer exists.")
        return

    vStore = load_vector_store_local(record['vectorPath'], embeddingModel)
    llm = get_gemini_llm(config["googleApiKey"])

    if vStore and llm:
        print(f"\n🔄 Resuming Chat: {record['folderName']}")
        chat_loop(db, record, vStore, llm, config, s_choice)

def main():
    config = load_app_configuration()
    embed_model = get_embedding_model()
    db = connect_to_mongodb(config) 

    while True:
        print("\n=== Gemini RAG with Memory ===")
        print("1. Process Static Folders (Scan PDF/Docs)")
        print("2. Start New Chat")
        print("3. Resume Previous Session (List IDs)")
        print("4. Quit")
        act = input("Select: ")
        
        if act == '1': process_static_directory(db, embed_model, config)
        elif act == '2': start_new_chat(db, embed_model, config)
        elif act == '3': resume_previous_session(db, embed_model, config)
        elif act == '4': break

if __name__ == "__main__":
    main()