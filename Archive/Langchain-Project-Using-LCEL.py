import os
import uuid
import json
import datetime
import warnings
import logging

warnings.filterwarnings("ignore", message=".*LangSmith now uses UUID v7.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
logging.getLogger("langsmith").setLevel(logging.ERROR)

from dotenv import load_dotenv
from pymongo import MongoClient, errors

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_mongodb import MongoDBChatMessageHistory

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


from langchain_community.utilities import SerpAPIWrapper


# Configuration & Initialization

def load_app_configuration():
    """
    Loads environment variables (.env) and returns config dict.
    Also ensures vector store root directory exists.
    """
    load_dotenv()

    appConfig = {
        "MongoUrl": os.getenv("MONGO_URL"),
        "GoogleApiKey": os.getenv("GOOGLE_API_KEY"),
        "SerpApiKey": os.getenv("SERPAPI_API_KEY"),
        "DbName": os.getenv("DB_NAME", "RagFolderDb"),
        "CollectionName": os.getenv("COLLECTION_NAME", "FolderCollection"),
        "ChatCollectionName": "ChatHistories",
        "SessionMetaCollection": "SessionMetadata",
        "SourceDirectory": "./static",
        "VectorStoreRoot": "./vector_stores",
        "ChunkSize": 1000,
        "ChunkOverlap": 100
    }

    if not appConfig["MongoUrl"]:
        print(" MONGO_URL missing in .env")
        exit()

    if not appConfig["GoogleApiKey"]:
        print(" GOOGLE_API_KEY missing in .env")
        exit()

 
    if not os.path.exists(appConfig["VectorStoreRoot"]):
        os.makedirs(appConfig["VectorStoreRoot"])

    return appConfig


def connect_to_mongodb(appConfig):
    """
    Connects to MongoDB and returns database object.
    """
    try:
        client = MongoClient(appConfig["MongoUrl"], serverSelectionTimeoutMS=5000)
        client.server_info()
        print("MongoDB Connected")
        return client[appConfig["DbName"]]
    except errors.ServerSelectionTimeoutError:
        print("Failed to connect to MongoDB. Check URL.")
        exit()

#Session Metadata Functions

def save_session_metadata(mongoDatabase, sessionId, folderId, folderName, currentSummary=""):
    """
    Stores or updates metadata including the running Summary.
    """
    try:
        collection = mongoDatabase["SessionMetadata"]
        
        update_data = {
            "SessionId": sessionId,
            "FolderId": folderId,
            "FolderName": folderName,
            "LastActive": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

        # Only update summary if it's not empty
        if currentSummary:
            update_data["Summary"] = currentSummary

        collection.update_one(
            {"SessionId": sessionId},
            {"$set": update_data},
            upsert=True
        )
    except Exception as e:
        print(f"Error saving session metadata: {e}")

def fetch_all_sessions(mongoDatabase):
    try:
        return mongoDatabase["SessionMetadata"].find().sort("LastActive", -1)
    except Exception as e:
        print(f"Error fetching sessions: {e}")
        return []


def fetch_session_details(mongoDatabase, sessionId):
    try:
        return mongoDatabase["SessionMetadata"].find_one({"SessionId": sessionId})
    except Exception as e:
        print(f"  Error fetching session details: {e}")
        return None

# Folder Management Functions

def check_folder_exists_in_db(mongoDatabase, folderName, collectionName):
    """
    Returns True if folder already processed.
    """
    try:
        return mongoDatabase[collectionName].find_one({"FolderName": folderName}) is not None
    except Exception as e:
        print(f" Error checking folder existence: {e}")
        return False


def insert_folder_record(mongoDatabase, folderName, folderPath, vectorPath, tokenCount, collectionName):
    """
    Stores a processed folder record containing:
    - folder path
    - vector store path
    - token count
    """
    record = {
        "_id": str(uuid.uuid4()),
        "FolderName": folderName,
        "FolderPath": os.path.abspath(folderPath),
        "VectorPath": os.path.abspath(vectorPath),
        "TokenCount": tokenCount,
        "CreatedAt": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }

    try:
        mongoDatabase[collectionName].insert_one(record)
        print(f"📁 Saved metadata for folder '{folderName}'")
    except Exception as e:
        print(f"  Error inserting folder record: {e}")


def fetch_all_folders(mongoDatabase, collectionName):
    try:
        return mongoDatabase[collectionName].find({}, {"_id": 1, "FolderName": 1})
    except Exception as e:
        print(f"  Error fetching folders: {e}")
        return []


def fetch_folder_by_id(mongoDatabase, folderId, collectionName):
    try:
        return mongoDatabase[collectionName].find_one({"_id": folderId})
    except Exception as e:
        print(f"  Error fetching folder by id: {e}")
        return None


# Document Loading / Splitting / Embedding

def count_tokens(text):
    """Rough token estimator (1 token ≈ 4 chars)."""
    try:
        return len(text) // 4
    except:
        return 0


def get_embedding_model():
    """Loads HuggingFace sentence-transformers embedding model."""
    try:
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        print(f"  Embedding load error: {e}")
        return None


def get_gemini_llm(apiKey, model="gemini-2.5-flash", temperature=0.25):
    """Loads Gemini model."""
    try:
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=apiKey,
            temperature=temperature
        )
    except Exception as e:
        print(f"  Gemini initialization error: {e}")
        return None


def load_documents_from_folder(folderPath):
    """
    Loads PDF, TXT, DOCX, CSV using appropriate loaders.
    """
    allDocs = []

    loader_map = {
        "**/*.pdf": (PyPDFLoader, {}),
        "**/*.txt": (TextLoader, {"encoding": "utf-8"}),
        "**/*.docx": (UnstructuredWordDocumentLoader, {}),
        "**/*.csv": (CSVLoader, {"encoding": "utf-8"})
    }

    try:
        for pattern, (loader_cls, kwargs) in loader_map.items():
            loader = DirectoryLoader(
                folderPath,
                glob=pattern,
                loader_cls=loader_cls,
                loader_kwargs=kwargs,
                silent_errors=True
            )
            docs = loader.load()
            if docs:
                allDocs.extend(docs)

        return allDocs

    except Exception as e:
        print(f"  Error loading documents: {e}")
        return []


def split_documents(documents, appConfig):
    """
    Splits loaded docs into chunks.
    """
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=appConfig["ChunkSize"],
            chunk_overlap=appConfig["ChunkOverlap"]
        )
        return splitter.split_documents(documents)
    except Exception as e:
        print(f"  Error splitting documents: {e}")
        return []


def create_vector_store(docChunks, embeddingModel):
    """
    Creates FAISS vector store.
    """
    try:
        return FAISS.from_documents(docChunks, embeddingModel)
    except Exception as e:
        print(f"  Error creating vector store: {e}")
        return None


def save_vector_store(vectorStore, appConfig):
    """
    Saves vector store to local directory.
    """
    try:
        folderId = str(uuid.uuid4())
        savePath = os.path.join(appConfig["VectorStoreRoot"], folderId)
        vectorStore.save_local(savePath)
        return savePath
    except Exception as e:
        print(f"  Error saving vector store: {e}")
        return None


def process_static_directory(mongoDatabase, embeddingModel, appConfig):
    """
    Scans ./static/, processes each subfolder into vector store.
    """
    sourceDir = appConfig["SourceDirectory"]
    if not os.path.exists(sourceDir):
        print(f" Source directory '{sourceDir}' missing.")
        return

    subFolders = [f.path for f in os.scandir(sourceDir) if f.is_dir()]
    print(f"\nFound {len(subFolders)} folders to process.\n")

    for folderPath in subFolders:
        folderName = os.path.basename(folderPath)

        if check_folder_exists_in_db(mongoDatabase, folderName, appConfig["CollectionName"]):
            print(f"Skipping '{folderName}' (already processed).")
            continue

        print(f"\n Processing folder: {folderName}")

        docs = load_documents_from_folder(folderPath)
        if not docs:
            print(f"  No documents found in {folderName}")
            continue

        chunks = split_documents(docs, appConfig)
        vectorStore = create_vector_store(chunks, embeddingModel)

        if vectorStore:
            savePath = save_vector_store(vectorStore, appConfig)
            tokenCount = count_tokens(" ".join([c.page_content for c in chunks]))

            insert_folder_record(
                mongoDatabase,
                folderName,
                folderPath,
                savePath,
                tokenCount,
                appConfig["CollectionName"]
            )

    print("\n Completed batch folder processing.\n")


# UNIFIED SINGLE-CALL RAG PROMPT + CHAIN

def get_unified_prompt():
    """
    This prompt does EVERYTHING in ONE LLM CALL:
    """
    return ChatPromptTemplate.from_template("""
You are a helpful RAG assistant. Follow these rules:

RULES:
1. For questions about CONVERSATION HISTORY (like "what did I ask", "summarize our chat", "list my questions", "create summary of questions"):
   - Use ONLY the chat_history_summary to answer.
   - This is NOT a RAG failure - you CAN answer these from history.
   - Provide a well-formatted summary or list.

2. For questions about DOCUMENT CONTENT:
   - Use the retrieved document context to answer.

3. ONLY set final_answer to "RAG_CANNOT_ANSWER" if:
   - The question is about document content AND context doesn't have the answer
   - AND the question is NOT about conversation history

4. Never fabricate information.
5. ALWAYS update the history summary to include the current Q&A.

FORMATTING RULES:
- Use proper spacing and line breaks in your answers.
- Use bullet points or numbered lists when listing items.
- Keep answers clear, organized, and easy to read.
- When summarizing conversation history, list each topic/question clearly with numbers.

SUMMARY RULES:
- The new_history_summary MUST include ALL previous topics from chat_history_summary.
- Add the current question and a brief note about the answer.
- Keep summary concise but complete (<= 150 words).
- Format: "Topics discussed: 1) topic1, 2) topic2, 3) topic3..."

INPUTS:
--------------------------
Chat History Summary:
{chat_history_summary}

--------------------------
Retrieved Document Context:
{context}

--------------------------
User Question:
{input}

--------------------------

Return STRICT VALID JSON ONLY (no markdown code blocks):

{{
  "final_answer": "Your well-formatted answer here",
  "new_history_summary": "Complete updated summary including all previous + current Q&A"
}}
""")




def build_rag(vectorStore, llm):
    """
    Builds the unified 1-call RAG chain:
    Retrieval → Prompt → LLM
    Returns both the chain and the retriever for source tracking.
    """
    try:
        retriever = vectorStore.as_retriever(search_kwargs={"k": 5})

        prompt = get_unified_prompt()

        # Extract the query string for the retriever, not the whole dict
        def retrieve_context(x):
            query = x["input"] if isinstance(x, dict) else x
            docs = retriever.invoke(query)
            return "\n\n".join([doc.page_content for doc in docs])

        chain = (
            {
                "context": retrieve_context,
                "input": lambda x: x["input"],
                "chat_history_summary": lambda x: x["chat_history_summary"],
            }
            | prompt
            | llm
        )

        return chain, retriever

    except Exception as e:
        print(f"  Error building RAG chain: {e}")
        return None, None


def perform_web_search_fallback(userQuery, llm, appConfig):
    """
    Fallback when RAG fails:
    1. SerpAPI search (external API call)
    2. LLM synthesizes an answer from search results (SECOND LLM CALL)

    Returns tuple: (answer_text, sources_list)
    sources_list contains dicts with 'title' and 'url' keys
    """
    import re
    
    try:
        if not appConfig["SerpApiKey"]:
            return "Web search disabled (missing SERPAPI_API_KEY).", []

        web_sources = []
        serpData = ""
        
        # Try using google-search-results package directly for better source extraction
        try:
            from serpapi import GoogleSearch
            
            params = {
                "q": userQuery,
                "api_key": appConfig["SerpApiKey"],
                "num": 5
            }
            search = GoogleSearch(params)
            results = search.get_dict()
            
            search_snippets = []
            
            # Get organic search results
            organic_results = results.get("organic_results", [])
            for result in organic_results[:5]:
                title = result.get("title", "Unknown")
                link = result.get("link", "")
                snippet = result.get("snippet", "")
                
                if link:
                    web_sources.append({"title": title, "url": link})
                if snippet:
                    search_snippets.append(f"From {title}: {snippet}")
            
            # Check answer box if available
            if "answer_box" in results:
                ab = results["answer_box"]
                if "answer" in ab:
                    search_snippets.insert(0, f"Quick Answer: {ab['answer']}")
                elif "snippet" in ab:
                    search_snippets.insert(0, f"Quick Answer: {ab['snippet']}")
                if "link" in ab:
                    web_sources.insert(0, {"title": ab.get("title", "Answer Box"), "url": ab["link"]})
            
            # Check knowledge graph
            if "knowledge_graph" in results:
                kg = results["knowledge_graph"]
                if "description" in kg:
                    search_snippets.insert(0, f"Overview: {kg['description']}")
                if "website" in kg:
                    web_sources.insert(0, {"title": kg.get("title", "Knowledge Graph"), "url": kg["website"]})
            
            serpData = "\n\n".join(search_snippets) if search_snippets else "No relevant results found."
            
        except ImportError:
            # Fallback to langchain SerpAPIWrapper
            try:
                search = SerpAPIWrapper(serpapi_api_key=appConfig["SerpApiKey"])
                serpData = search.run(userQuery)
                
                # Try to extract URLs from the text response
                urls = re.findall(r'https?://[^\s<>"\)\]]+', serpData)
                unique_urls = list(dict.fromkeys(urls))[:5]
                web_sources = [{"title": f"Web Result {i+1}", "url": url} for i, url in enumerate(unique_urls)]
                
            except Exception as e:
                return f"Web search failed: {e}", []
                
        except Exception as e:
            # Fallback to langchain SerpAPIWrapper on any error
            try:
                search = SerpAPIWrapper(serpapi_api_key=appConfig["SerpApiKey"])
                serpData = search.run(userQuery)
                
                urls = re.findall(r'https?://[^\s<>"\)\]]+', serpData)
                unique_urls = list(dict.fromkeys(urls))[:5]
                web_sources = [{"title": f"Web Result {i+1}", "url": url} for i, url in enumerate(unique_urls)]
                
            except Exception as e2:
                return f"Web search failed: {e2}", []

        # LLM synthesizes from web data
        prompt = f"""User asked: {userQuery}

Local document knowledge was insufficient.
Here is the WEB SEARCH RESULT:
{serpData}

Create the best possible answer using ONLY this web data.
Format your answer clearly with proper spacing and structure.
If the web data is not useful, say:
"I could not find a reliable answer even after searching the web."
"""

        try:
            response = llm.invoke(prompt)
            return response.content.strip(), web_sources
        except Exception as e:
            return f"Gemini failed to interpret search results: {e}", web_sources

    except Exception as e:
        return f"Unexpected web search error: {e}", []


def load_vector_store_local(vectorPath, embeddingModel):
    """
    Loads a FAISS index stored at a given directory.
    """
    try:
        return FAISS.load_local(
            vectorPath,
            embeddingModel,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"  Error loading vector store: {e}")
        return None


def get_history(appConfig):
    """
    Returns a callable that gives MongoDBChatMessageHistory instance
    for a given session.
    """
    def create(sessionId):
        try:
            return MongoDBChatMessageHistory(
                session_id=sessionId,
                connection_string=appConfig["MongoUrl"],
                database_name=appConfig["DbName"],
                collection_name=appConfig["ChatCollectionName"]
            )
        except Exception as e:
            print(f"  Error connecting chat history: {e}")
            return None

    return create


def format_document_sources(docs):
    """
    Extracts and formats source information from retrieved documents.
    Returns a formatted string with file names and page numbers.
    """
    sources = []
    seen = set()
    
    for doc in docs:
        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
        
        
        source = metadata.get('source', metadata.get('file_path', 'Unknown'))
        
        if source and source != 'Unknown':
            source = os.path.basename(source)
       
        page = metadata.get('page', metadata.get('page_number', None))
        
        
        if page is not None:
            key = f"{source}:p{page}"
            display = f"{source} (Page {int(page) + 1})"
        else:
            key = source
            display = f"{source}"
        
        if key not in seen:
            seen.add(key)
            sources.append(display)
    
    return sources

# MAIN CHAT LOOP

def chat_loop(mongoDatabase, dbRecord, vectorStore, llm, appConfig, sessionId, useWebSearch, initial_metadata=None):
    try:
        ragChain, retriever = build_rag(vectorStore, llm)
        if ragChain is None: return

        # Initialize Summary from metadata if resuming, else empty
        current_chat_summary = ""
        if initial_metadata and "Summary" in initial_metadata:
            current_chat_summary = initial_metadata["Summary"]
            print(f"Loaded Context: {current_chat_summary[:100]}...")

        # Initial save to establish session
        save_session_metadata(mongoDatabase, sessionId, dbRecord["_id"], dbRecord["FolderName"], current_chat_summary)

        print(f"\n Chat Started for folder: {dbRecord['FolderName']}")
        print("Type 'exit' to quit.\n")

        sessionHistory = get_history(appConfig)(sessionId)
        if sessionHistory is None:
            print("Failed to initialize chat history.")
            return
        
        while True:
            userQuery = input("You: ").strip()
            if userQuery.lower() == "exit": break
            if not userQuery: continue

            print("Thinking...", end="\r")

            try:
                # 1. RUN RAG CHAIN
                # We pass current_chat_summary directly into the chain input
                rawResponse = ragChain.invoke({
                    "input": userQuery,
                    "chat_history_summary": current_chat_summary 
                })

                # --- Response Parsing Logic (Same as before) ---
                response_text = ""
                if hasattr(rawResponse, 'content'): response_text = rawResponse.content
                elif isinstance(rawResponse, dict): response_text = rawResponse.get('content', str(rawResponse))
                else: response_text = str(rawResponse)
                
                # Clean JSON markdown
                response_text = response_text.strip()
                if response_text.startswith("```json"): response_text = response_text.split("```json", 1)[1]
                if response_text.endswith("```"): response_text = response_text.rsplit("```", 1)[0]
                
                result = json.loads(response_text.strip())
                # -----------------------------------------------

                # 2. HANDLE RAG FAILURE (WEB SEARCH)
                if result.get("final_answer") == "RAG_CANNOT_ANSWER":
                    if useWebSearch:
                        print("\n RAG could not answer. Using Web Search...")
                        webAnswer, webSources = perform_web_search_fallback(userQuery, llm, appConfig)
                        
                        print(f"\nAnswer: {webAnswer}")
                        
                        # Update Summary manually for web search
                        current_chat_summary = f"{current_chat_summary} | Web Q&A: {userQuery} -> Answered via web."
                        
                        # Save to History
                        sessionHistory.add_user_message(userQuery)
                        sessionHistory.add_ai_message(webAnswer)
                        
                        # Update Metadata
                        save_session_metadata(mongoDatabase, sessionId, dbRecord["_id"], dbRecord["FolderName"], current_chat_summary)
                        continue
                    else:
                        print("\n I cannot answer this question from the local knowledge base.")
                        continue

                # 3. HANDLE SUCCESS
                final_answer = result.get("final_answer", "")
                new_summary = result.get("new_history_summary", "")

                print(f"\n Answer:\n{final_answer}")
                
                # 4. UPDATE STATE
                # Update the variable for the next loop iteration
                if new_summary:
                    current_chat_summary = new_summary

                # Add Standard Messages to History (Human + AI only)
                sessionHistory.add_user_message(userQuery)
                sessionHistory.add_ai_message(final_answer)

                # 5. SAVE METADATA (Hidden from Chat History)
                # We save the summary to the separate Metadata collection here
                save_session_metadata(mongoDatabase, sessionId, dbRecord["_id"], dbRecord["FolderName"], current_chat_summary)

            except Exception as e:
                print(f"\n  Error processing query: {e}")

        print("\n Chat ended.")

    except Exception as e:
        print(f" Critical failure in chat loop: {e}")

def start_new_chat(mongoDatabase, embeddingModel, appConfig):
    """
    Displays all folders → user selects 1 → chat begins.
    """
    try:
        folders = list(fetch_all_folders(mongoDatabase, appConfig["CollectionName"]))
        if not folders:
            print("  No processed folders found. Run option 1 first.")
            return

        print("\nAvailable Knowledge Bases:\n")
        print(f"{'ID':<38} | Folder Name")
        print("-" * 60)
        for f in folders:
            print(f"{f['_id']:<38} | {f['FolderName']}")

        chosenId = input("\nEnter Folder ID: ").strip()
        useWeb = input("Enable Web Search fallback? (y/n): ").strip().lower() == "y"

        folderRecord = fetch_folder_by_id(mongoDatabase, chosenId, appConfig["CollectionName"])
        if not folderRecord:
            print(" Invalid folder ID.")
            return

        vectorStore = load_vector_store_local(folderRecord["VectorPath"], embeddingModel)
        if vectorStore is None:
            print(" Failed to load vector store.")
            return

        llm = get_gemini_llm(appConfig["GoogleApiKey"], model="gemini-2.5-flash", temperature=0.25)
        if llm is None:
            print("Failed to initialize LLM.")
            return

        newSessionId = str(uuid.uuid4())
        chat_loop(mongoDatabase, folderRecord, vectorStore, llm, appConfig, newSessionId, useWeb)

    except Exception as e:
        print(f"  Error starting new chat: {e}")


def resume_previous_session(mongoDatabase, embeddingModel, appConfig):
    try:
        sessions = list(fetch_all_sessions(mongoDatabase))
        if not sessions:
            print("\n  No saved sessions found.")
            return

        print("\nPrevious Sessions:\n")
        print(f"{'Session ID':<38} | {'Folder':<20} | Last Active")
        print("-" * 85)
        for s in sessions:
            sid = str(s.get("SessionId", "Unknown"))
            fname = str(s.get("FolderName", "Unknown"))
            raw_date = s.get("LastActive", "-")
            la = raw_date[:16].replace("T", " ") if isinstance(raw_date, str) else str(raw_date)[:16]
            print(f"{sid:<38} | {fname:<20} | {la}")

        chosenId = input("\nEnter Session ID: ").strip()
        if not chosenId: return

        useWeb = input("Enable Web Search fallback? (y/n): ").strip().lower() == "y"

        # Fetch metadata to get FolderId AND the stored Summary
        meta = fetch_session_details(mongoDatabase, chosenId)
        if not meta:
            print("Session not found.")
            return

        folderRecord = fetch_folder_by_id(mongoDatabase, meta["FolderId"], appConfig["CollectionName"])
        if not folderRecord:
            print(" Folder for this session no longer exists.")
            return

        vectorStore = load_vector_store_local(folderRecord["VectorPath"], embeddingModel)
        if vectorStore is None:
            print(" Failed to load vector store.")
            return

        llm = get_gemini_llm(appConfig["GoogleApiKey"])
        if llm is None: return

        print(f"\n Resuming chat: {folderRecord['FolderName']}")
        
        # Pass the loaded metadata (which contains the 'Summary') to the chat loop
        chat_loop(mongoDatabase, folderRecord, vectorStore, llm, appConfig, chosenId, useWeb, initial_metadata=meta)

    except Exception as e:
        print(f"  Error resuming session: {e}")
def main():
   
    print("      ---Gemini RAG System (Full)---       ")
   
    appConfig = load_app_configuration()
    mongoDatabase = connect_to_mongodb(appConfig)
    embeddingModel = get_embedding_model()

    while True:
        print("\n========= MENU =========")
        print("1. Process Folder Contents (PDF/TXT/DOCX/CSV)")
        print("2. Start New Chat")
        print("3. Resume Previous Session")
        print("4. Quit")
        print("========================")

        choice = input("Select option: ").strip()

        if choice == "1":
            process_static_directory(mongoDatabase, embeddingModel, appConfig)
        elif choice == "2":
            start_new_chat(mongoDatabase, embeddingModel, appConfig)
        elif choice == "3":
            resume_previous_session(mongoDatabase, embeddingModel, appConfig)
        elif choice == "4":
            print("\n Exiting. Goodbye!")
            break
        else:
            print(" Invalid option. Try again.")

if __name__ == "__main__":
    main()

