import os
import datetime
import uuid
import warnings
import logging

# Suppress warnings
warnings.filterwarnings("ignore", message=".*LangSmith now uses UUID v7.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
logging.getLogger("langsmith").setLevel(logging.ERROR)

# Environment & Database
from dotenv import load_dotenv
from pymongo import MongoClient, errors
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_mongodb import MongoDBChatMessageHistory

#Embeddings & Vector Store
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, DirectoryLoader, CSVLoader
)
#Agent & Tooling
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents import create_tool_calling_agent
from langchain_community.utilities import SerpAPIWrapper

#Configuration & MongoDB Connection

def load_app_configuration():
    """Loads configuration from .env and sets up necessary directories."""
    try:
        load_dotenv()
        
        appConfig = {
            "MongoUrl": os.getenv("mongo_url"),
            "GoogleApiKey": os.getenv("google_api_key"),
            "SerpApiKey": os.getenv("serpapi_api_key"),
            "DbName": os.getenv("db_name"),
            "CollectionName": os.getenv("collection_name"),
            "ChatCollectionName": os.getenv("chat_collection_name"),
            "SessionMetaCollection": os.getenv("session_meta_collection"),
            "SourceDirectory": os.getenv("source_directory"),
            "VectorStoreRoot": os.getenv("vector_store_root"),
            "ChunkSize": int(os.getenv("chunk_size",1000)),
            "ChunkOverlap": int(os.getenv("chunk_overlap",100)),
            "LlmModel": os.getenv("llm_model"),
            "LlmTemperature": float(os.getenv("llm_temperature",0.3)),
            "RetrieverK": int(os.getenv("retriever_k",5)),
            "AgentMaxIterations": int(os.getenv("agent_max_iterations",5)),
            "AgentVerbose": os.getenv("agent_verbose", "true").lower() == "true"
        }

        if not appConfig["MongoUrl"] or not appConfig["GoogleApiKey"]:
            print("Error: Missing API Keys in .env file.")
            exit()

        if not os.path.exists(appConfig["VectorStoreRoot"]):
            os.makedirs(appConfig["VectorStoreRoot"])

        return appConfig
    except Exception as e:
        print(f"Error loading configuration: {e}")
        exit()

def connect_to_mongodb(appConfig):
    try:
        mongoClient = MongoClient(appConfig["MongoUrl"], serverSelectionTimeoutMS=5000)
        mongoClient.server_info() 
        return mongoClient[appConfig["DbName"]]
    except errors.ServerSelectionTimeoutError:
        print("Failed to connect to MongoDB. Check your URL.")
        exit()
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
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
    try:
        uniqueId = str(uuid.uuid4())
        
        dbRecord = {
            "_id": uniqueId,
            "FolderName": folderName,
            "FolderPath": os.path.abspath(folderPath),
            "VectorPath": os.path.abspath(vectorPath),
            "TokenCount": tokenCount,
            "CreatedAt": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
        
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

def get_gemini_llm(apiKey,model,temperature):
    """Returns a Gemini LLM instance."""
    try:
        return ChatGoogleGenerativeAI(model=model, google_api_key=apiKey, temperature=temperature)
    except Exception as e:
        print(f"Error getting Gemini LLM: {e}")
        return None

def load_documents_from_folder(folderPath):
    """Loads documents from a folder using various loaders."""
    try:
        allDocuments = []
        loaderMapping = {
            "**/*.pdf":  (PyPDFLoader, {}),
            "**/*.txt":  (TextLoader, {'encoding': 'utf-8'}),
            "**/*.docx": (UnstructuredWordDocumentLoader, {}),
            "**/*.csv":  (CSVLoader, {'encoding': 'utf-8'}),
            "**/*.md":   (TextLoader, {'encoding': 'utf-8'})
        }
        
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
    try:
        sourceDir = appConfig["SourceDirectory"]
        if not os.path.exists(sourceDir):
            print(f"Source dir '{sourceDir}' not found.")
            return
        
        subFolders = [f.path for f in os.scandir(sourceDir) if f.is_dir()]
        print(f"Found {len(subFolders)} folders. Processing...")
        
        for subFolder in subFolders:
            try:
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
            except Exception as e:
                print(f"Error processing folder {subFolder}: {e}")
                continue
        
        print("Batch processing complete.")
    except Exception as e:
        print(f"Error in process_static_directory: {e}")


# Conversational RAG Chain with Memory

def load_vector_store_local(vectorPath, embeddingModel):
    try:
        return FAISS.load_local(vectorPath, embeddingModel, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None

class RagSearchInput(BaseModel):
    query: str = Field(..., 
                       description="The specific question or keywords to search for in the local document database.")

class WebSearchInput(BaseModel):
    query: str = Field(..., 
                       description="The search query for finding information on the internet.")


# Tool Implementations

def create_web_search_tool(appConfig, enabled=True):
    """Creates the web search tool using SerpAPI."""
    
    @tool("web_search", args_schema=WebSearchInput)
    def web_search_tool(query: str) -> str:
        """
        Search the internet for Skyscape or Buzz App related information 
        ONLY if it is not found in the local documents.
        RESTRICT SEARCH TO: Buzz App, Skyscape, Healthcare Communication.
        """
        try:
            # Check if web search is enabled
            if not enabled:
                return "Web search is disabled for this session. Only document search is available."
            
            apiKey = appConfig.get("SerpApiKey")
            if not apiKey:
                return "Web search disabled (missing SERPAPI_API_KEY)."
            
            serpApi = SerpAPIWrapper(serpapi_api_key=apiKey)
            # Append context to query to ensure relevance
            contextualQuery = f"{query} Skyscape Buzz App"
            rawResults = serpApi.results(contextualQuery)
            
            formattedOutput = []
            
            if "answer_box" in rawResults:
                answerBox = rawResults["answer_box"]
                if "answer" in answerBox:
                    formattedOutput.append(f"Quick Answer: {answerBox['answer']}")
                elif "snippet" in answerBox:
                    formattedOutput.append(f"Quick Answer: {answerBox['snippet']}")
            
            if "knowledge_graph" in rawResults:
                knowledgeGraph = rawResults["knowledge_graph"]
                if "description" in knowledgeGraph:
                    formattedOutput.append(f"Description: {knowledgeGraph['description']}")
            
            if "organic_results" in rawResults:
                formattedOutput.append("\nWeb Results:")
                for i, result in enumerate(rawResults["organic_results"][:5], 1):
                    title = result.get("title", "No title")
                    snippet = result.get("snippet", "No description")
                    link = result.get("link", "")
                    formattedOutput.append(f"\n[{i}] {title}")
                    formattedOutput.append(f"    {snippet}")
                    formattedOutput.append(f"    Link: {link}")
            
            if formattedOutput:
                return "\n".join(formattedOutput)
            else:
                return serpApi.run(contextualQuery)
                
        except Exception as e:
            return f"Error during web search: {e}"
    
    return web_search_tool


def create_rag_search_tool(retriever):
    """Creates the RAG document search tool using the given retriever."""
    
    @tool("document_search", args_schema=RagSearchInput)
    def rag_search_tool(query: str) -> str:
        """
        Search for information within the uploaded documents and knowledge base.
        Use this FIRST for any questions about the document content.
        Returns relevant passages from the documents.
        """
        try:
            docs = retriever.invoke(query)
            if not docs:
                return "No relevant information found in the documents."
            
            results = []
            for i, doc in enumerate(docs, 1):
                source = os.path.basename(doc.metadata.get("source", "unknown"))
                page = doc.metadata.get("page", "N/A")
                content = doc.page_content[:500]  
                results.append(f"[{i}] Source: {source} (Page {page})\n{content}")
            
            return "\n\n".join(results)
        except Exception as e:
            return f"Error searching documents: {e}"
    
    return rag_search_tool


#MongoDB Chat History 

def get_mongodb_chat_history(appConfig, sessionId):
    """Returns MongoDB chat history for the given session."""
    try:
        return MongoDBChatMessageHistory(
            session_id=sessionId,
            connection_string=appConfig["MongoUrl"],
            database_name=appConfig["DbName"],
            collection_name=appConfig["ChatCollectionName"]
        )
    except Exception as e:
        print(f"Error connecting to chat history: {e}")
        return None


def load_history_as_messages(chatHistory):
    """Loads chat history as a list of messages for the agent."""
    messages = []
    try:
        if chatHistory and hasattr(chatHistory, 'messages'):
            for msg in chatHistory.messages:
                messages.append(msg)
    except Exception as e:
        print(f"Error loading history: {e}")
    return messages


def format_agent_response(response):
    """
    Cleans and formats the agent response.
    Handles cases where response is a list of dicts with 'type' and 'text' fields.
    """
    try:
        if response is None:
            return "No response generated."
        
        
        if isinstance(response, str):
            if response.startswith("[{") and "'type': 'text'" in response:
                try:
                    import ast
                    parsed = ast.literal_eval(response)
                    if isinstance(parsed, list):
                        texts = []
                        for item in parsed:
                            if isinstance(item, dict) and 'text' in item:
                                texts.append(item['text'])
                            elif isinstance(item, str):
                                texts.append(item)
                        return " ".join(texts).strip()
                except:
                    pass
            return response
        
       
        if isinstance(response, list):
            texts = []
            for item in response:
                if isinstance(item, dict):
                    if 'text' in item:
                        texts.append(item['text'])
                elif isinstance(item, str):
                    texts.append(item)
            return " ".join(texts).strip() if texts else str(response)

        if isinstance(response, dict) and 'text' in response:
            return response['text']
        
        
        return str(response)
    except Exception as e:
        print(f"Error formatting response: {e}")
        return str(response)

def build_agent_prompt(webSearchEnabled):
    """Builds the system prompt for the agent based on configuration."""
    try:
        # Dynamic instruction for web search based on availability
        if webSearchEnabled:
            webInstruction = """2. **web_search** – SECONDARY FALLBACK  
   Use ONLY if:  
   - document_search returns nothing OR  
   - the requested Buzz/Skyscape information is not found in any document.
   - **CRITICAL:** If `document_search` returns "not found" or "documents do not state...", you MUST use `web_search` to find the answer (provided it is about Buzz/Skyscape).
   When using web_search, restrict your answer to **Buzz / Skyscape / healthcare communication context only.**  
   NEVER answer unrelated general knowledge questions."""
        else:
            webInstruction = """2. **web_search** – DISABLED  
   This tool is currently disabled. Do not use it. Rely only on document_search."""

        systemPrompt = f"""You are BuzzBot — a helpful AI assistant specialized in answering questions ONLY about:

- The Buzz Application
- Buzz Organizations
- Buzz Features, Settings, How-to Guides
- Skyscape products (Buzz, BuzzLink, BuzzVideo, Lightning, etc.)

A comprehensive internal document has been provided.  
**This document contains details of the Buzz app. Any Buzz-related question should be answered from this document first.**

You have access to the following tools:

1. **document_search** – PRIMARY SOURCE  
   Use this tool first for every question.  
   Retrieve relevant information from the Buzz knowledge base documents.

{webInstruction}

---

# RULES FOR ALL RESPONSES

### 📌 1. Priority of Information Sources  
- Always search documents first via **document_search**.  
- Use **web_search ONLY as fallback** when absolutely necessary (and if enabled).
- **CRITICAL:** If `document_search` yields no results or says the info is missing, AND the question is about Buzz/Skyscape, you MUST try `web_search` before giving up.
- Never hallucinate answers. If not found in docs OR web → say so politely.

### 📌 2. Topic Restriction  
The bot is STRICTLY LIMITED to answering:  
- Buzz features  
- Buzz usage  
- BuzzPhone  
- Buzz Organizations  
- Buzz Communication  
- Skyscape products integrated with Buzz  
- **Content of the Knowledge Base:** You ARE allowed to answer questions about what is in the documents/folder (e.g., "What does this file contain?", "Summarize the documents").

**Handling Off-Topic Queries:**
If the user asks **anything outside Buzz/Skyscape** (e.g., "Who won the match?", "Python code", "Cooking recipes"), reply:  
“Sorry, I can only answer questions related to Buzz or Skyscape products.”  
(do NOT call web_search for unrelated queries)

**Ambiguous Terms:**
If a user asks about a general term (e.g., "Groups", "Chats", "Contacts"), ALWAYS interpret it in the context of the Buzz Application.
If a user asks "What is in this folder?" or "What do you know?", interpret it as asking about the **Buzz Knowledge Base**.

### 3. Comparison Questions (e.g., X vs Y)  
- Use **document_search** for items available in Buzz docs.  
- Use **web_search** ONLY for items NOT in the documents.  
- Combine both sources and provide a clean comparison.

###  4. Meta Questions  
If the user asks about the conversation (e.g., “What did I ask earlier?”, "Summarize our chat"),  
you MUST look at the **chat_history**.  
Do NOT call any tools for meta questions.

### 📌 5. Citation Rules  
- **MANDATORY:** You MUST cite your source for every answer.
- **Placement:** Citations must appear on a **separate line** below the relevant text or bullet point.
- **Format for Documents:**
  - PDF/DOCX: `Source: [Document Name] (Page [X])`
  - Markdown/Text/JSON: `Source: [Document Name]`
- **Format for Web:**
  - `Source: Web Search - [Link]`
- **Combined Answers:** Group citations clearly at the bottom if they apply to the whole response, or under each specific point if mixed.

### 📌 6. Style  
- Responses must be clean, structured, and concise.  
- No unnecessary text.  
- No speculation.
"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", systemPrompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        return prompt
    except Exception as e:
        print(f"Error building agent prompt: {e}")
        return None


def create_agent_executor(llm, vectorStore, appConfig, webSearchEnabled=True):
    """
    Creates an Agent Executor
    """
    try:
        # Create retriever from vector store
        retriever = vectorStore.as_retriever(search_kwargs={"k": 5})
        
        # Create tools
        ragTool = create_rag_search_tool(retriever)
        webTool = create_web_search_tool(appConfig, enabled=webSearchEnabled)
        
        tools = [ragTool, webTool]
        
        # Build the agent prompt
        prompt = build_agent_prompt(webSearchEnabled)
        if prompt is None:
            print("Error: Failed to build agent prompt.")
            return None
        
        # Create the agent
        agent = create_tool_calling_agent(llm, tools, prompt)
        
        # Create executor 
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True, 
            handle_parsing_errors=True,
            max_iterations=5
        )
        
        return executor
    except Exception as e:
        print(f"Error creating agent executor: {e}")
        import traceback
        traceback.print_exc()
        return None

# Main Chat Loop

def chat_loop(mongoDatabase, dbRecord, vectorStore, llm, appConfig, sessionId, webSearchEnabled=None):
    """Main chat loop with MongoDB-backed memory using Agent."""
    try:
       
        if webSearchEnabled is None:
            print("\n Web Search Option:")
            print("   [Y] Enable web search (search internet if document doesn't have answer)")
            print("   [N] Disable web search (only search documents)")
            webChoice = input("Enable web search? (Y/n): ").strip().lower()
            webSearchEnabled = webChoice != 'n'
        
        
        agent_executor = create_agent_executor(llm, vectorStore, appConfig, webSearchEnabled)
        
        if agent_executor is None:
            print("Error: Failed to create agent executor.")
            return
        
       
        chatHistory = get_mongodb_chat_history(appConfig, sessionId)
        if chatHistory is None:
            print("Error: Failed to connect to chat history.")
            return
        
        # Load existing history as messages
        historyMessages = load_history_as_messages(chatHistory)
        
        
        save_session_metadata(mongoDatabase, sessionId, dbRecord["_id"], dbRecord["FolderName"])
        
        webStatus = "Enabled" if webSearchEnabled else "Disabled"
        print(f"\nChat Started | Folder: {dbRecord['FolderName']}")
        print(f"Session ID: {sessionId}")
        print(f"Web Search: {webStatus}")
        print(f"Loaded {len(historyMessages)} previous messages")
        print("Type 'exit' to quit.\n")
        
        while True:
            try:
                userQuery = input("\nYou: ").strip()
                if userQuery.lower() == 'exit':
                    break
                if not userQuery:
                    continue
                
                print("Thinking...", end="\r")
                
                try:
                    
                    result = agent_executor.invoke({
                        "input": userQuery,
                        "chat_history": historyMessages
                    })
                    
                    # Get the answer and clean it
                    rawAnswer = result.get("output", "No response generated.")
                    answer = format_agent_response(rawAnswer)
                    
                    print(f"\nAssistant:\n{answer}")
                    
                    # Save to MongoDB chat history
                    chatHistory.add_user_message(userQuery)
                    chatHistory.add_ai_message(answer)
                    
                    # Update local history for next iteration
                    historyMessages.append(HumanMessage(content=userQuery))
                    historyMessages.append(AIMessage(content=answer))
                    
                    # Update session metadata
                    save_session_metadata(mongoDatabase, sessionId, dbRecord["_id"], dbRecord["FolderName"])
                    
                except Exception as e:
                    print(f"\nError processing request: {e}")
                    import traceback
                    traceback.print_exc()
            except KeyboardInterrupt:
                print("\n\nChat interrupted by user.")
                break
            except Exception as e:
                print(f"\nError in chat loop: {e}")
                continue
        
        print("\nChat ended. History saved to MongoDB.")
    except Exception as e:
        print(f"Error in chat_loop: {e}")
        import traceback
        traceback.print_exc()

# User Interaction Functions

def start_new_chat(mongoDatabase, embeddingModel, appConfig):
    """Starts a new chat session."""
    try:
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
            llm = get_gemini_llm(appConfig["GoogleApiKey"], model="gemini-2.5-flash", temperature=0.3)
            
            if vectorStore and llm:
                newSessionId = str(uuid.uuid4())
                chat_loop(mongoDatabase, folderRecord, vectorStore, llm, appConfig, newSessionId)
            else:
                print("Error loading vector store or LLM.")
        else:
            print("Invalid ID.")
    except Exception as e:
        print(f"Error starting new chat: {e}")

def resume_previous_session(mongoDatabase, embeddingModel, appConfig):
    """Resumes a previous chat session."""
    try:
        sessions = list(fetch_all_sessions(mongoDatabase))
        if not sessions:
            print("\nNo saved session history found yet. Start a new chat first!")
            return

        print(f"\n{'Session ID':<38} | {'Folder Name':<20} | {'Last Active'}")
        print("-" * 80)
        
        
        for s in sessions:
            sId = s.get('SessionId', 'Unknown')
            fName = s.get('FolderName', 'Unknown')
            lActive = s.get('LastActive', '')[:16].replace('T', ' ')
            print(f"{sId:<38} | {fName[:20]:<20} | {lActive}")

        sessionChoice = input("\nEnter Session ID to resume (or 'b' to back): ").strip()
        if sessionChoice.lower() == 'b':
            return

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
        llm = get_gemini_llm(appConfig["GoogleApiKey"], model="gemini-2.5-flash", temperature=0.3)

        if vectorStore and llm:
            print(f"\nResuming Chat: {folderRecord['FolderName']}")
            chat_loop(mongoDatabase, folderRecord, vectorStore, llm, appConfig, sessionChoice)
        else:
            print("Error loading vector store or LLM.")
    except Exception as e:
        print(f"Error resuming session: {e}")

def main():
    try:
        appConfig = load_app_configuration()
        embedModel = get_embedding_model()
        mongoDatabase = connect_to_mongodb(appConfig) 

        while True:
            try:
                print("\n=== QnA Bot ===")
                print("1. Process Static Folders (Scan PDF/Docs)")
                print("2. Start New Chat")
                print("3. Resume Previous Session")
                print("4. Quit")
                userAction = input("Select: ")
                
                match userAction:
                    case '1': 
                        process_static_directory(mongoDatabase, embedModel, appConfig)
                    case '2': 
                        start_new_chat(mongoDatabase, embedModel, appConfig)
                    case '3': 
                        resume_previous_session(mongoDatabase, embedModel, appConfig)
                    case '4': 
                        break
                    case _:
                        print("Invalid selection.")
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"Error in main menu: {e}")
                continue
    except Exception as e:
        print(f"Fatal error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()