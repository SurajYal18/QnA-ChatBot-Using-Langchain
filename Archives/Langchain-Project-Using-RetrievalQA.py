import os
import re
import datetime
import uuid
import shutil
from dotenv import load_dotenv
from pymongo import MongoClient, errors

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    UnstructuredWordDocumentLoader, 
    DirectoryLoader,
    CSVLoader
)
from langchain_core.documents import Document

# Configuration
def load_app_configuration():
    """
    Loads environment variables and sets up application configuration.
    """
    load_dotenv()

    mongoUrl = os.getenv("MONGO_URL")
    googleApiKey = os.getenv("GOOGLE_API_KEY")

    if not mongoUrl:
        print("Error: MONGO_URL not found in .env file.")
        exit()
    
    if not googleApiKey:
        print("Error: GOOGLE_API_KEY not found in .env file.")
        exit()

    config = {
        "mongoUrl": mongoUrl,
        "googleApiKey": googleApiKey,
        "dbName": os.getenv("DB_NAME", "rag_folder_db"),
        "collectionName": "Simple_store",
        "sourceDirectory": "./static",        
        "vectorStoreRoot": "./vector_stores", 
        "chunkSize": 1000, 
        "chunkOverlap": 100
    }

    if not os.path.exists(config["vectorStoreRoot"]):
        os.makedirs(config["vectorStoreRoot"])

    return config


# Database Functions 
def connect_to_mongodb(config):
    """Establishes connection to MongoDB using config dict."""
    try:
        client = MongoClient(config["mongoUrl"], serverSelectionTimeoutMS=5000)
        client.server_info()
        database = client[config["dbName"]]
        collection = database[config["collectionName"]]
        print("Connected to MongoDB successfully.")
        return collection
    except errors.ServerSelectionTimeoutError:
        print("Failed to connect to MongoDB. Check your .env file.")
        exit()

def check_folder_exists_in_db(dbCollection, folderName):
    try:
        record = dbCollection.find_one({"folderName": folderName})
        return record is not None
    except Exception as e:
        print(f"Error checking folder in DB: {e}")

def insert_folder_record(dbCollection, folderName, folderPath, vectorPath, tokenCount):
    """Inserts metadata about the processed folder into MongoDB."""
    uniqueId = str(uuid.uuid4())
    currentTime = datetime.datetime.now(datetime.timezone.utc).isoformat()

    record = {
        "_id": uniqueId,
        "folderName": folderName,
        "folderPath": os.path.abspath(folderPath),
        "vectorPath": os.path.abspath(vectorPath),
        "tokenCount": tokenCount,
        "createdAt": currentTime
    }
    
    try:
        dbCollection.insert_one(record)
        print(f"Database record created for: {folderName}")
        return True
    except Exception as e:
        print(f"Error inserting record: {e}")
        return False

def fetch_all_folders(dbCollection):
    """Retrieves all folder records from MongoDB."""
    try:
        return dbCollection.find({}, {"_id": 1, "folderName": 1, "createdAt": 1, "tokenCount": 1})
    except Exception as e:
        print(f"Error fetching folders: {e}")
        return []

def fetch_folder_by_id(dbCollection, folderId):
    """Retrieves a specific folder record by its UUID."""
    try:
        return dbCollection.find_one({"_id": folderId})
    except Exception as e:
        print(f"Error fetching folder by ID: {e}")


# Core Processing Functions
def count_tokens(textInput):
    """Simple token counter based on character length approximation."""
    try:
        if not textInput: return 0
        return len(textInput) // 4
    except Exception:
        return 0

def beautify_text(rawText):
    """Cleans and formats raw text for better readability."""
    cleanText = re.sub(r'([a-zA-Z])([A-Z])', r'\1 \2', rawText)
    cleanText = re.sub(r'\s{2,}', ' ', cleanText)
    cleanText = cleanText.replace("●", "\n• ")
    return cleanText.strip()

def get_embedding_model():
    """Initializes and returns the HuggingFace embedding model."""
    try:
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        print(f"Error getting embeddings: {e}")
        raise e

def get_gemini_llm(apiKey):
    """Initializes and returns the Gemini LLM."""
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=apiKey,
            temperature=0.3
        )
        return llm
    except Exception as e:
        print(f"Error initializing Gemini: {e}")
        return None

# Document Processing Functions
def load_documents_from_folder(folderPath):
    """
    Loads documents dynamically based on file extension mapping.
    """
    allDocuments = []
   
    loader_mapping = {
        "**/*.pdf":  (PyPDFLoader, {}),
        "**/*.txt":  (TextLoader, {'encoding': 'utf-8'}),
        "**/*.docx": (UnstructuredWordDocumentLoader, {}),
        "**/*.csv":  (CSVLoader, {'encoding': 'utf-8'}),
    }

    try:
        for glob_pattern, (loader_cls, loader_kwargs) in loader_mapping.items():
            loader = DirectoryLoader(
                folderPath, 
                glob=glob_pattern, 
                silent_errors=True # Optional: skips files that fail to load
            )
            loaded_docs = loader.load()
            if loaded_docs:
                allDocuments.extend(loaded_docs)
                
        return allDocuments
    except Exception as e:
        print(f"Error loading files in {folderPath}: {e}")
        return []

def split_documents(documents, config):
    """Splits loaded documents into smaller chunks using config settings."""
    textSplitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunkSize"], 
        chunk_overlap=config["chunkOverlap"]
    )
    return textSplitter.split_documents(documents)

def create_vector_store(docChunks, emeddingModel):
    """Generates the FAISS vector store in memory from document chunks."""
    try:
        vectorStore = FAISS.from_documents(docChunks, emeddingModel)
        return vectorStore
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None
    
def save_vector_store(vectorStore, config):
    """Saves the FAISS vector store to a unique local directory."""
    uniqueFolderId = str(uuid.uuid4())
    savePath = os.path.join(config["vectorStoreRoot"], uniqueFolderId)

    try:
        vectorStore.save_local(savePath)
        return savePath
    except Exception as e:
        print(f"Error saving vector store: {e}")
        return None
    
def process_single_folder_workflow(folderPath, dbCollection, embeddingModel, config):
    """Processing of a single folder."""
    folderName = os.path.basename(os.path.normpath(folderPath))

    if check_folder_exists_in_db(dbCollection, folderName):
        print(f"Skipping '{folderName}': Already exists in database.")
        return

    print(f"Processing folder: {folderName}...")

    documents = load_documents_from_folder(folderPath)
    if not documents:
        print(f"No supported documents found in {folderName}.")
        return
    
    print(f"   - Loaded {len(documents)} pages/rows.")

    docChunks = split_documents(documents, config)
    if not docChunks:
        print("   - Documents were empty or could not be split.")
        return

    fullText = " ".join([d.page_content for d in docChunks])
    totalTokens = count_tokens(fullText)

    vectorStore = create_vector_store(docChunks, embeddingModel)

    if vectorStore:
        vectorSavePath = save_vector_store(vectorStore, config)

        if vectorSavePath:
            insert_folder_record(dbCollection, folderName, folderPath, vectorSavePath, totalTokens)

def process_static_directory(dbCollection, embeddingModel, config):
    """Processes all subfolders in the static source directory."""
    sourceDir = config["sourceDirectory"]

    if not os.path.exists(sourceDir):
        print(f"Source directory '{sourceDir}' not found.")
        return

    subFolders = [f.path for f in os.scandir(sourceDir) if f.is_dir()]

    if not subFolders:
        print(f"No subfolders found inside '{sourceDir}'.")
        return

    print(f"Found {len(subFolders)} folders in '{sourceDir}'. Starting batch process...")
    print("="*50)

    for subFolder in subFolders:
        process_single_folder_workflow(subFolder, dbCollection, embeddingModel, config)
    
    print("="*50)
    print("Batch processing complete.")


# Chat functions

def load_vector_store_local(vectorPath, embeddingModel):
    """Loads a FAISS vector store from a local directory."""
    try:
        return FAISS.load_local(
            vectorPath,
            embeddingModel,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"Critical Error loading vector store: {e}")
        return None

def create_rag_chain(vectorStore, llm):
    """Configures the Prompt and the Retrieval Chain."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the provided context, just say, "answer is not available in the context".

    Context:
    {context}

    Question: 
    {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    retriever = vectorStore.as_retriever(search_kwargs={"k": 5})
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

def display_response(result):
    """Handles the pretty printing of the answer and sources."""
    answer = result["result"]
    source_docs = result["source_documents"]

    print("\n✨ Gemini Answer:")
    print(answer)
    
    print("\n--- Sources Used ---")
    seen_sources = set()
    for doc in source_docs:
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page", "N/A")
        row = doc.metadata.get("row", "N/A")

        if source not in seen_sources:
            loc_info = f"Pg {page}" if page != "N/A" else f"Row {row}"
            print(f"📄 {source} ({loc_info})")
            seen_sources.add(source)
    print("-" * 50)

def chat_loop(dbRecord, vectorStore, llm):
    """Main user input loop."""
    print(f"\n💬 Agent Active for: {dbRecord['folderName']} (Type 'exit' to stop)\n")
    
    # 1. Initialize Chain once
    qa_chain = create_rag_chain(vectorStore, llm)

    # 2. Start Loop
    while True:                       
        userQuery = input(" \nQuestion: ").strip()
        if userQuery.lower() == 'exit':
            break

        print("Thinking...", end="\r")

        try:
            result = qa_chain.invoke({"query": userQuery})
            display_response(result)
        except Exception as e:
            print(f"Error generating response: {e}")
            
def display_folders_and_select(dbCollection, embeddingModel, config):
    """Displays folders from DB and allows user to select one for chatting."""
    cursor = fetch_all_folders(dbCollection)
    folders = list(cursor)
    
    if not folders:
        print("\n No folders found in database.")
        return

    print(f"\n{'ID':<38} | {'Folder Name':<20} | {'Tokens':<10} | {'Created'}")
    print("-" * 85)
    
    for doc in folders:
        folderId = str(doc.get('_id', 'Unknown'))
        folderName = doc.get('folderName', 'Unknown Name')
        tokenCount = doc.get('tokenCount', 0)
        createdAt = doc.get('createdAt', 'N/A')

        print(f"{folderId:<38} | {folderName[:20]:<20} | "
              f"{tokenCount:<10} | {createdAt[:10]}")

    userChoice = input("\n Enter Folder ID (UUID) to chat (or 'b' to back): ").strip()
    
    if userChoice.lower() == 'b':
        return

    dbRecord = fetch_folder_by_id(dbCollection, userChoice)

    if not dbRecord:
        print("Folder ID not found.")
        return

    print(f"\n Loading: {dbRecord.get('folderName', 'Unknown')}...")
    
    vectorPath = dbRecord.get('vectorPath')
    if not vectorPath:
        print("Error: No vector path found in record.")
        return

    vectorStore = load_vector_store_local(vectorPath, embeddingModel)
    llm = get_gemini_llm(config["googleApiKey"])

    if vectorStore and llm:
        chat_loop(dbRecord, vectorStore, llm)


def main(): 
    """Main entry point for the application."""
    # 1. Load Config
    appConfig = load_app_configuration()
    
    print(" Initializing Embedding Model...")
    embeddingModel = get_embedding_model()
    
    print(" Connecting to Database...")
    dbCollection = connect_to_mongodb(appConfig)

    while True:
        print("\n" + "=" * 50)
        print("Gemini RAG: Intelligent Document Chat")
        print("=" * 50)
        print("1. Process All Folders (PDF, DOCX, CSV)")
        print("2. Select a Folder to Chat")
        print("3. Quit")

        actionInput = input("\nSelect Action: ")

        if actionInput == '1':
            process_static_directory(dbCollection, embeddingModel, appConfig)
        elif actionInput == '2':
            display_folders_and_select(dbCollection, embeddingModel, appConfig)
        elif actionInput == '3':
            print("👋 Goodbye!")
            break
        else:
            print("Invalid selection.")

if __name__ == "__main__":
    main()