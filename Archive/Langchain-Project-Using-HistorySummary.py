import os
import datetime
import uuid
import warnings
import logging

# Suppress warnings
warnings.filterwarnings("ignore", message=".*LangSmith now uses UUID v7.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
logging.getLogger("langsmith").setLevel(logging.ERROR)

from dotenv import load_dotenv
from pymongo import MongoClient, errors

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_mongodb import MongoDBChatMessageHistory

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, DirectoryLoader, CSVLoader
)


#Configuration & MongoDB Connection

def load_app_configuration():
    """Loads configuration from .env and sets up necessary directories."""
    load_dotenv()
    
    appConfig = {
        "MongoUrl": os.getenv("MONGO_URL"),
        "GoogleApiKey": os.getenv("GOOGLE_API_KEY"),
        "DbName": os.getenv("DB_NAME", "RagFolderDb"),
        "CollectionName": os.getenv("COLLECTION_NAME", "DabCollection"),
        "ChatCollectionName": "ChatHistories",
        "SessionMetaCollection": "SessionMetadata",
        "SourceDirectory": "./static",
        "VectorStoreRoot": "./vector_stores",
        "ChunkSize": 1000,
        "ChunkOverlap": 100
    }

    if not appConfig["MongoUrl"] or not appConfig["GoogleApiKey"]:
        print("Error: Missing API Keys in .env file.")
        exit()

    if not os.path.exists(appConfig["VectorStoreRoot"]):
        os.makedirs(appConfig["VectorStoreRoot"])

    return appConfig

def connect_to_mongodb(appConfig):
    try:
        mongoClient = MongoClient(appConfig["MongoUrl"], serverSelectionTimeoutMS=5000)
        mongoClient.server_info() 
        return mongoClient[appConfig["DbName"]]
    except errors.ServerSelectionTimeoutError:
        print("Failed to connect to MongoDB. Check your URL.")
        exit()


#Session Meetadata

def save_session_metadata(mongoDatabase, sessionId, folderId, folderName):
    """Saves or updates session metadata in MongoDB."""
    try:
            metaCollection = mongoDatabase["SessionMetadata"]
            
            metaCollection.update_one(
                {"SessionId": sessionId},
                {
                    "$set": {
                        "SessionId": sessionId,
                        "FolderId": str(folderId),
                        "FolderName": folderName,
                        "LastActive": datetime.datetime.now(datetime.timezone.utc).isoformat()
                    }
                },
                upsert=True
            )
    except Exception as e:
        print(f"Error saving session metadata: {e}")

def fetch_all_sessions(mongoDatabase):
    """Retrieves all session metadata from MongoDB."""
    try:
        metaCollection = mongoDatabase["SessionMetadata"]
        allSessions = metaCollection.find().sort("LastActive", -1)
        return allSessions
    except Exception as e:
        print(f"Error fetching all sessions: {e}")
        return []

def fetch_session_details(mongoDatabase, sessionId):
    """Fetches specific session metadata by SessionId."""
    try:
        metaCollection = mongoDatabase["SessionMetadata"]
        return metaCollection.find_one({"SessionId": sessionId})
    except Exception as e:
        print(f"Error fetching session details: {e}")
        return None


# Folder Records Management

def check_folder_exists_in_db(mongoDatabase, folderName, collectionName):
    """Checks if a folder with the given name exists in the database."""
    try:
       return mongoDatabase[collectionName].find_one({"FolderName": folderName}) is not None
    except Exception as e:
        print(f"Error accessing collection: {e}")
        return False
    

def insert_folder_record(mongoDatabase, folderName, folderPath, vectorPath, tokenCount, collectionName):
    """Inserts a new folder record into the database."""
    uniqueId = str(uuid.uuid4())
    
    dbRecord = {
        "_id": uniqueId,
        "FolderName": folderName,
        "FolderPath": os.path.abspath(folderPath),
        "VectorPath": os.path.abspath(vectorPath),
        "TokenCount": tokenCount,
        "CreatedAt": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }
    
    try:
        mongoDatabase[collectionName].insert_one(dbRecord)
        print(f"Database record created for: {folderName}")
    except Exception as e:
        print(f"Error inserting record: {e}")

def fetch_all_folders(mongoDatabase, collectionName):
    """Fetches all folder records from the database."""
    try:
        return mongoDatabase[collectionName].find({}, {"_id": 1, "FolderName": 1})
    except Exception as e:
        print(f"Error fetching all folders: {e}")
        return []

def fetch_folder_by_id(mongoDatabase, folderId, collectionName):
    """Fetches a specific folder record by its ID."""
    try:
        return mongoDatabase[collectionName].find_one({"_id": folderId})
    except Exception as e:
        print(f"Error fetching folder by ID: {e}")
        return None

#Core Document Processing & Vector Store Creation

def count_tokens(textInput):
    """Counts tokens in a text input."""
    try:
        return len(textInput) // 4 if textInput else 0
    except Exception as e:
        print(f"Error counting tokens: {e}")
        return 0

def get_embedding_model():
    """Returns an embedding model."""
    try:
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        print(f"Error getting embedding model: {e}")
        return None

def get_gemini_llm(apiKey, model="gemini-2.5-flash", temperature=0.3):
    """Returns a Gemini LLM instance with configurable model and temperature."""
    try:
        return ChatGoogleGenerativeAI(
            model=model, 
            google_api_key=apiKey, 
            temperature=temperature
        )
    except Exception as e:
        print(f"Error getting Gemini LLM: {e}")
        return None

def load_documents_from_folder(folderPath):
    """Loads documents from a folder using various loaders."""
    allDocuments = []
    loaderMapping = {
        "**/*.pdf":  (PyPDFLoader, {}),
        "**/*.txt":  (TextLoader, {'encoding': 'utf-8'}),
        "**/*.docx": (UnstructuredWordDocumentLoader, {}),
        "**/*.csv":  (CSVLoader, {'encoding': 'utf-8'}),
    }
    try:
        for globPattern, (loaderCls, loaderKwargs) in loaderMapping.items():
            dirLoader = DirectoryLoader(folderPath, glob=globPattern, loader_cls=loaderCls, loader_kwargs=loaderKwargs, silent_errors=True)
            loadedDocs = dirLoader.load()
            if loadedDocs: 
                allDocuments.extend(loadedDocs)
        return allDocuments
    except Exception as e:
        print(f"Error loading files: {e}")
        return []

def split_documents(documents, appConfig):
    """Splits documents into chunks."""
    try:
        textSplitter = RecursiveCharacterTextSplitter(
            chunk_size=appConfig["ChunkSize"], 
            chunk_overlap=appConfig["ChunkOverlap"]
        )
        return textSplitter.split_documents(documents)
    except Exception as e:
        print(f"Error splitting documents: {e}")
        return []

def create_vector_store(docChunks, embeddingModel):
    """Creates a FAISS vector store from document chunks."""
    try:
        return FAISS.from_documents(docChunks, embeddingModel)
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None

def save_vector_store(vectorStore, appConfig):
    """Saves the vector store locally and returns the path."""
    try:
        uniqueFolderId = str(uuid.uuid4())
        savePath = os.path.join(appConfig["VectorStoreRoot"], uniqueFolderId)
        vectorStore.save_local(savePath)
        return savePath
    except Exception as e:
        print(f"Error saving vector store: {e}")
        return None

def process_static_directory(mongoDatabase, embeddingModel, appConfig):
    """Processes all subfolders in the source directory."""
    sourceDir = appConfig["SourceDirectory"]
    if not os.path.exists(sourceDir):
        print(f"Source dir '{sourceDir}' not found.")
        return
    
    subFolders = [f.path for f in os.scandir(sourceDir) if f.is_dir()]
    print(f"Found {len(subFolders)} folders. Processing...")
    
    for subFolder in subFolders:
        folderName = os.path.basename(os.path.normpath(subFolder))
        
        if check_folder_exists_in_db(mongoDatabase, folderName, appConfig["CollectionName"]):
            print(f"Skipping '{folderName}' (Exists).")
            continue
            
        print(f"Processing: {folderName}...")
        documents = load_documents_from_folder(subFolder)
        
        if not documents: 
            continue
            
        docChunks = split_documents(documents, appConfig)
        vectorStore = create_vector_store(docChunks, embeddingModel)
        
        if vectorStore:
            vectorSavePath = save_vector_store(vectorStore, appConfig)
            totalTokens = count_tokens(" ".join([d.page_content for d in docChunks]))
            
            insert_folder_record(
                mongoDatabase, 
                folderName, 
                subFolder, 
                vectorSavePath, 
                totalTokens,
                appConfig["CollectionName"]
            )
    print("Batch processing complete.")


# Conversational RAG Chain with Memory

def load_vector_store_local(vectorPath, embeddingModel):
    try:
        return FAISS.load_local(vectorPath, embeddingModel, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None

def summarize_chat_history(chatHistory, llm):
    """
    CHANGE 1: New function to summarize chat history using LLM.
    This reduces the context size and makes retrieval more efficient.
    Only keeps last 2 exchanges raw, summarizes older messages.
    """
    if len(chatHistory) <= 4:  # Keep last 2 Q&A pairs (4 messages) as-is
        return chatHistory
    
    # Split: older messages to summarize, recent messages to keep
    messagesToSummarize = chatHistory[:-4]
    recentMessages = chatHistory[-4:]
    
    # Build conversation text from older messages
    conversationText = ""
    for msg in messagesToSummarize:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        conversationText += f"{role}: {msg.content}\n"
    
    # Create summarization prompt
    summaryPrompt = f"""Summarize the following conversation history concisely, 
capturing key points, questions asked, and answers provided. Keep it under 200 words.

Conversation:
{conversationText}

Summary:"""
    
    try:
        # Call LLM to generate summary
        summaryResponse = llm.invoke(summaryPrompt)
        summary = summaryResponse.content
        
        # Create summarized history: [summary] + recent messages
        summarizedHistory = [
            HumanMessage(content=f"[Previous conversation summary: {summary}]")
        ] + recentMessages
        
        return summarizedHistory
    except Exception as e:
        print(f"Error summarizing chat history: {e}")
        return chatHistory  # Return original if summarization fails

def get_qa_prompt():
    """Returns a QA prompt template."""
    try:
        systemPrompt = (
            "You are an assistant for question-answering tasks. "
            "Use ONLY the following pieces of retrieved context to answer the question. "
            "If the context does not contain information to answer the question, "
            "you MUST respond with: "
            "'I cannot answer this question as it's not covered in the available documents. "
            "This knowledge base contains information about [briefly mention what topics are in context]. "
            "Please ask questions related to these topics.'\n\n"
            "CRITICAL RULES:\n"
            "1. NEVER use your general knowledge - ONLY use the provided context\n"
            "2. If context is irrelevant to the question, say you don't know\n"
            "3. Do NOT make up information or hallucinate answers\n"
            "4. Check chat history only for clarification of previous questions\n\n"
            "Context:\n{context}"
        )
        return ChatPromptTemplate.from_messages([
            ("system", systemPrompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
    except Exception as e:
        print(f"Error creating QA prompt: {e}")
        return None

def get_history(appConfig):
    """Returns a callable that retrieves the specific session history from Mongo."""
    def get_session_history(sessionId: str) -> BaseChatMessageHistory:
        return MongoDBChatMessageHistory(
            session_id=sessionId,
            connection_string=appConfig["MongoUrl"],
            database_name=appConfig["DbName"],
            collection_name=appConfig["ChatCollectionName"]
        )
    return get_session_history

def build_rag_chain(vectorStore, llm, appConfig):
    """
    CHANGE 3: Simplified RAG chain - removed history-aware retriever.
    Now uses direct retrieval + QA chain with summarized history.
    This reduces LLM calls from 3 to 2:
    - Call 1: Summarize chat history (only when history > 4 messages)
    - Call 2: Answer question using context + summarized history
    """
    try:
        # Simple retriever without history-aware contextualization
        retriever = vectorStore.as_retriever(search_kwargs={"k": 5})

        qaPrompt = get_qa_prompt()
        if qaPrompt is None:
            print("Error: QA prompt failed to load.")
            return None

        # Direct QA chain without contextualization step
        questionAnswerChain = create_stuff_documents_chain(llm, qaPrompt)
        
        ragChain = create_retrieval_chain(retriever, questionAnswerChain)
        
        # Wrap with message history
        conversationalRagChain = RunnableWithMessageHistory(
            ragChain,
            get_history(appConfig), 
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        
        return conversationalRagChain
    except Exception as e:
        print(f"Error building RAG chain: {e}")
        return None

def check_context_relevance(userQuery, retrievedDocs, llm):
    """
    NEW FUNCTION: Validates if retrieved documents are relevant to the question.
    Returns True if relevant, False if not.
    """
    if not retrievedDocs:
        return False
    
    # Build context preview (first 500 chars of each doc)
    contextPreview = "\n".join([
        f"Doc {i+1}: {doc.page_content[:500]}..."
        for i, doc in enumerate(retrievedDocs[:3])
    ])
    
    relevancePrompt = f"""Question: {userQuery}

Retrieved Context Preview:
{contextPreview}

Is the retrieved context relevant and sufficient to answer this question?
Answer with only "YES" or "NO" and a brief reason (one sentence).

Format: YES/NO - [reason]"""
    
    try:
        response = llm.invoke(relevancePrompt)
        answer = response.content.strip().upper()
        return answer.startswith("YES")
    except Exception as e:
        print(f"Error checking relevance: {e}")
        return True  # Default to allowing if check fails
    
def display_response(result, folderName):
    """
    CHANGE 4: Enhanced display with folder context and better error messages.
    Displays the answer and sources with page numbers.
    """
    print("\n✨ Gemini Answer:")
    print(result["answer"])
    
    # Check if answer indicates unavailable information
    if "cannot answer" in result["answer"].lower() or "don't know" in result["answer"].lower():
        print(f"\n💡 Tip: This knowledge base focuses on '{folderName}'.")
        print("   Try asking questions related to this topic.")
        return
    
    if "context" in result and result["context"]:
        print("\n--- Sources Used ---")
        
        # Dictionary to group pages by filename
        # Structure: { "filename.pdf": {1, 3, 5} }
        source_map = {}
        
        for doc in result["context"]:
            src = os.path.basename(doc.metadata.get("source", "unknown"))
            page = doc.metadata.get("page") # PyPDFLoader saves this
            
            if src not in source_map:
                source_map[src] = set()
            
            # Check if page exists (Text/CSV files won't have page numbers)
            if page is not None:
                # Add 1 because LangChain uses 0-based indexing (Page 1 is index 0)
                source_map[src].add(page + 1)
        
        # Display the grouped results
        for filename, pages in source_map.items():
            if pages:
                # Sort pages numerically: 1, 2, 5
                sorted_pages = sorted(list(pages))
                page_str = ", ".join(map(str, sorted_pages))
                print(f"📄 {filename} (Pages: {page_str})")
            else:
                # Fallback for files without pages (like .txt)
                print(f"📄 {filename}")

        print("-" * 50)
        
def chat_loop(mongoDatabase, dbRecord, vectorStore, llm, appConfig, sessionId):
    """
    CHANGE 5: Added relevance checking before generating answers.
    Main chat loop for user interaction.
    """
    ragChain = build_rag_chain(vectorStore, llm, appConfig)

    if ragChain is None:
        print("Error: Failed to build RAG chain.")
        return

    save_session_metadata(mongoDatabase, sessionId, dbRecord["_id"], dbRecord["FolderName"])

    print(f"\n💬 Chat Started | Topic: {dbRecord['FolderName']}")
    print(f"🔑 Session ID: {sessionId}")
    print("Type 'exit' to stop.\n")
    
    while True:
        userQuery = input(" \nQuestion (or 'exit'): ").strip()
        if userQuery.lower() == 'exit': break
        if not userQuery: continue

        print("Thinking...", end="\r")
        
        try:
            # Retrieve current chat history
            sessionHistory = get_history(appConfig)(sessionId)
            currentHistory = sessionHistory.messages
            
            # Summarize chat history if it's getting long
            summarizedHistory = summarize_chat_history(currentHistory, llm)
            
            # Temporarily replace history with summarized version
            sessionHistory.clear()
            for msg in summarizedHistory:
                sessionHistory.add_message(msg)
            
            # First, retrieve documents to check relevance
            retriever = vectorStore.as_retriever(search_kwargs={"k": 5})
            retrievedDocs = retriever.invoke(userQuery)
            
            # Check if retrieved context is relevant
            isRelevant = check_context_relevance(userQuery, retrievedDocs, llm)
            
            if not isRelevant:
                print(f"\n❌ Cannot Answer:")
                print(f"Your question is not related to '{dbRecord['FolderName']}'.")
                print(f"Please ask questions about the topics covered in this knowledge base.\n")
                continue
            
            # Invoke RAG chain with summarized history
            result = ragChain.invoke(
                {"input": userQuery},
                config={"configurable": {"session_id": sessionId}}
            )
          
            display_response(result, dbRecord['FolderName'])
            save_session_metadata(mongoDatabase, sessionId, dbRecord["_id"], dbRecord["FolderName"])
            
        except Exception as e:
            print(f"\nError processing request: {e}")

# User Interaction Menus

def start_new_chat(mongoDatabase, embeddingModel, appConfig):
    """Starts a new chat session."""
    folders = list(fetch_all_folders(mongoDatabase, appConfig["CollectionName"]))
    if not folders:
        print("No folders found. Run Option 1 first.")
        return

    print(f"\n{'ID':<38} | {'Folder Name':<20}")
    print("-" * 60)
    
    for doc in folders:
        print(f"{str(doc['_id']):<38} | {doc['FolderName'][:20]:<20}")
        
    userChoice = input("\nEnter Folder ID: ").strip()
    folderRecord = fetch_folder_by_id(mongoDatabase, userChoice, appConfig["CollectionName"])
    
    if folderRecord:
        vectorStore = load_vector_store_local(folderRecord['VectorPath'], embeddingModel)
        llm = get_gemini_llm(
            appConfig["GoogleApiKey"], 
            model="gemini-2.5-flash", 
            temperature=0.3
        )
        
        if vectorStore and llm:
            newSessionId = str(uuid.uuid4())
            chat_loop(mongoDatabase, folderRecord, vectorStore, llm, appConfig, newSessionId)
    else:
        print("Invalid ID.")

def resume_previous_session(mongoDatabase, embeddingModel, appConfig):
    """Resumes a previous chat session."""
    sessions = list(fetch_all_sessions(mongoDatabase))
    if not sessions:
        print("\n No saved session history found yet. Start a new chat first!")
        return

    print(f"\n{'Session ID':<38} | {'Folder Name':<20} | {'Last Active'}")
    print("-" * 80)
    
    for s in sessions:
        sId = s.get('SessionId', 'Unknown')
        fName = s.get('FolderName', 'Unknown')
        lActive = s.get('LastActive', '')[:16].replace('T', ' ')
        print(f"{sId:<38} | {fName[:20]:<20} | {lActive}")

    sessionChoice = input("\nEnter Session ID to resume (or 'b' to back): ").strip()
    if sessionChoice.lower() == 'b': return

    sessionMeta = fetch_session_details(mongoDatabase, sessionChoice)
    if not sessionMeta:
        print("Session ID not found.")
        return

    folderId = sessionMeta.get('FolderId')
    folderRecord = fetch_folder_by_id(mongoDatabase, folderId, appConfig["CollectionName"])
    
    if not folderRecord:
        print("Error: The folder associated with this session no longer exists.")
        return

    vectorStore = load_vector_store_local(folderRecord['VectorPath'], embeddingModel)
    llm = get_gemini_llm(
        appConfig["GoogleApiKey"], 
        model="gemini-2.5-flash", 
        temperature=0.3
    )

    if vectorStore and llm:
        print(f"\n🔄 Resuming Chat: {folderRecord['FolderName']}")
        chat_loop(mongoDatabase, folderRecord, vectorStore, llm, appConfig, sessionChoice)

def main():
    appConfig = load_app_configuration()
    embedModel = get_embedding_model()
    mongoDatabase = connect_to_mongodb(appConfig) 

    while True:
        print("\n=== Gemini RAG with Memory ===")
        print("1. Process Static Folders (Scan PDF/Docs)")
        print("2. Start New Chat")
        print("3. Resume Previous Session")
        print("4. Quit")
        userAction = input("Select: ")
        
        if userAction == '1': 
            process_static_directory(mongoDatabase, embedModel, appConfig)
        elif userAction == '2': 
            start_new_chat(mongoDatabase, embedModel, appConfig)
        elif userAction == '3': 
            resume_previous_session(mongoDatabase, embedModel, appConfig)
        elif userAction == '4': 
            break

if __name__ == "__main__":
    main()