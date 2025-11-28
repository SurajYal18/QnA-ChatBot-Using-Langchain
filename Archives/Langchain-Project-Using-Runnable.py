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
from langchain_core.messages import HumanMessage, AIMessage

from langchain_mongodb import MongoDBChatMessageHistory

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, DirectoryLoader, CSVLoader
)

from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.runnables import (
    RunnableSequence, RunnableParallel, RunnableLambda, 
    RunnablePassthrough, RunnableBranch
)
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import SerpAPIWrapper
import re
import json

#Configuration & MongoDB Connection

def load_app_configuration():
    """Loads configuration from .env and sets up necessary directories."""
    try:
        load_dotenv()
        
        appConfig = {
            "MongoUrl": os.getenv("MONGO_URL"),
            "GoogleApiKey": os.getenv("GOOGLE_API_KEY"),
            "SerpApiKey": os.getenv("SERPAPI_API_KEY"),
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


#Session Metadata

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




def create_web_search_tool(appConfig, enabled=True):
    """Creates the web search tool using SerpAPI."""
    
    @tool("web_search", args_schema=WebSearchInput)
    def web_search_tool(query: str) -> str:
        """
        Search the internet for current events, recent news, or information 
        not found in the local documents. Use this when document_search 
        doesn't have the answer.
        """
        try:
            
            if not enabled:
                return "Web search is disabled for this session. Only document search is available."
            
            api_key = appConfig.get("SerpApiKey")
            if not api_key:
                return "Web search disabled (missing SERPAPI_API_KEY)."
            
            serp_api = SerpAPIWrapper(serpapi_api_key=api_key)
            raw_results = serp_api.results(query)  # Get structured results
            
           
            info_parts = []
            sources = []
            
        
            if "answer_box" in raw_results:
                answer_box = raw_results["answer_box"]
                if "answer" in answer_box:
                    info_parts.append(answer_box['answer'])
                elif "snippet" in answer_box:
                    info_parts.append(answer_box['snippet'])
            
            
            if "knowledge_graph" in raw_results:
                kg = raw_results["knowledge_graph"]
                if "description" in kg:
                    info_parts.append(kg['description'])
                if "source" in kg and "link" in kg["source"]:
                    sources.append(kg["source"]["link"])
         
            if "organic_results" in raw_results:
                for result in raw_results["organic_results"][:5]:
                    snippet = result.get("snippet", "")
                    link = result.get("link", "")
                    if snippet:
                        info_parts.append(snippet)
                    if link:
                        sources.append(link)
            
            # Format output
            output = []
            if info_parts:
              
                unique_info = []
                for part in info_parts:
                    if part and part not in unique_info:
                        unique_info.append(part)
                output.append(" ".join(unique_info[:3]))  
            
            if sources:
                output.append("\nSource:")
                unique_sources = list(dict.fromkeys(sources[:5]))  
                for source in unique_sources:
                    output.append(f"*   {source}")
            
            if output:
                return "\n".join(output)
            else:
                
                return serp_api.run(query)
                
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
                content = doc.page_content[:500]  # Limit content length
                results.append(f"[{i}] Source: {source} (Page {page})\n{content}")
            
            return "\n\n".join(results)
        except Exception as e:
            return f"Error searching documents: {e}"
    
    return rag_search_tool

# ============================================
# MongoDB Chat History Helper
# ============================================

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


def load_history_as_messages(chat_history):
    """Loads chat history as a list of messages for the agent."""
    try:
        messages = []
        if chat_history and hasattr(chat_history, 'messages'):
            for msg in chat_history.messages:
                messages.append(msg)
        return messages
    except Exception as e:
        print(f"Error loading history: {e}")
        return []


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



def handle_meta_questions(inputs: dict) -> dict:
    """Handles meta-questions about conversation history."""
    try:
        query = inputs["input"].lower()
        chat_history = inputs.get("chat_history", [])
        
        meta_keywords = ["what did i ask", "previous question", "conversation history", 
                        "what we discussed", "earlier question"]
        
        if any(keyword in query for keyword in meta_keywords):
            if not chat_history:
                return {"type": "meta", "result": "No conversation history yet.", "query": inputs["input"]}
            
            user_questions = []
            for msg in chat_history:
                if hasattr(msg, 'type') and msg.type == 'human':
                    user_questions.append(msg.content)
            
            history = "\n".join([f"{i+1}. {q}" for i, q in enumerate(user_questions)])
            return {"type": "meta", "result": f"Previous questions:\n{history}", "query": inputs["input"]}
        
        return {"type": "query", "query": inputs["input"], "chat_history": chat_history}
    except Exception as e:
        print(f"Error in handle_meta_questions: {e}")
        return {"type": "query", "query": inputs.get("input", ""), "chat_history": inputs.get("chat_history", [])}


def create_search_documents_function(rag_tool, llm):
    """Creates the search_documents function with captured tools."""
    def search_documents(data: dict) -> dict:
        """Searches documents using RAG and validates relevance using LLM."""
        try:
            if data["type"] == "meta":
                return data
            
            query = data["query"]
            print("Searching documents...", end="\r")
            
            try:
                rag_result = rag_tool.invoke(query)
            except Exception as e:
                print(f"RAG tool error: {e}")
                return {
                    "type": "search_complete",
                    "query": query,
                    "rag_result": f"Error: {str(e)}",
                    "has_rag_content": False,
                    "chat_history": data.get("chat_history", [])
                }
            
            if "no relevant information" in rag_result.lower() or "error searching" in rag_result.lower():
                return {
                    "type": "search_complete",
                    "query": query,
                    "rag_result": rag_result,
                    "has_rag_content": False,
                    "chat_history": data.get("chat_history", [])
                }

            
            try:
                print("Verifying relevance...   ", end="\r")
                
                validation_prompt = f"""You are a relevance grader. 
User Question: {query}
Retrieved Documents:
{rag_result}

Does the retrieved information contain the answer to the user's question? 
Respond ONLY with 'YES' or 'NO'.
"""
                validation_response = llm.invoke(validation_prompt)
                content = validation_response.content if hasattr(validation_response, 'content') else str(validation_response)
                is_relevant = "YES" in content.strip().upper()
            except Exception as e:
                print(f"LLM validation error: {e}")
                is_relevant = False
            
            return {
                "type": "search_complete",
                "query": query,
                "rag_result": rag_result,
                "has_rag_content": is_relevant,
                "chat_history": data.get("chat_history", [])
            }
        except Exception as e:
            print(f"Error in search_documents: {e}")
            return {
                "type": "search_complete",
                "query": data.get("query", ""),
                "rag_result": f"Error: {str(e)}",
                "has_rag_content": False,
                "chat_history": data.get("chat_history", [])
            }
    
    return search_documents


def use_rag_only(data: dict) -> dict:
    """Uses only RAG results when sufficient."""
    try:
        print("Found in documents       ", end="\r")
        return {
            "type": "final",
            "query": data["query"],
            "sources": f"Document Search:\n{data['rag_result']}",
            "chat_history": data.get("chat_history", [])
        }
    except Exception as e:
        print(f"Error in use_rag_only: {e}")
        return {
            "type": "final",
            "query": data.get("query", ""),
            "sources": f"Error: {str(e)}",
            "chat_history": data.get("chat_history", [])
        }


def create_add_web_search_function(web_tool, web_search_enabled):
    """Creates the add_web_search function with captured tools."""
    def add_web_search(data: dict) -> dict:
        """Adds web search when RAG results are insufficient."""
        try:
            if not web_search_enabled:
                return use_rag_only(data)
            
            print("Searching web...         ", end="\r")
            
            try:
                web_result = web_tool.invoke(data["query"])
                combined = f"Document Search:\n{data['rag_result']}\n\nWeb Search:\n{web_result}"
                
                return {
                    "type": "final",
                    "query": data["query"],
                    "sources": combined,
                    "chat_history": data.get("chat_history", [])
                }
            except Exception as e:
                print(f"Web search error: {e}")
                return {
                    "type": "final",
                    "query": data["query"],
                    "sources": f"Document Search:\n{data['rag_result']}\n\nWeb Search: Error - {str(e)}",
                    "chat_history": data.get("chat_history", [])
                }
        except Exception as e:
            print(f"Error in add_web_search: {e}")
            return {
                "type": "final",
                "query": data.get("query", ""),
                "sources": f"Error: {str(e)}",
                "chat_history": data.get("chat_history", [])
            }
    
    return add_web_search


def format_final_prompt(data: dict) -> str:
    """Formats the final prompt for LLM."""
    try:
        system_prompt = """You are a helpful AI assistant with access to document search and web search.

**Your task:** Answer user questions clearly and accurately using the provided tool results.
**Always cite your sources** (document names for docs, URLs for web results).
"""
        
        if data["type"] == "meta":
            return f"{system_prompt}\n\nUser Question: {data['query']}\n\nAnswer: {data['result']}"
        
        query = data["query"]
        sources = data["sources"]
        chat_history = data.get("chat_history", [])
        
      
        has_document_sources = "Document Search:" in sources
        has_web_sources = "Web Search:" in sources
        
        
        history_text = ""
        if chat_history:
            items = []
            for msg in chat_history[-4:]:
                if hasattr(msg, 'type'):
                    role = "User" if msg.type == 'human' else "Assistant"
                    items.append(f"{role}: {msg.content[:100]}...")
            history_text = "\n".join(items)
        
        citation_instructions = ""
        if has_document_sources and not has_web_sources:
            citation_instructions = """
**Citation Format for Documents:**
- Use inline citations with numbers in square brackets [1], [2], etc.
- At the end of your response, list the sources:

Source:
*   [1] DocumentName.pdf, Page X
*   [2] DocumentName.pdf, Page Y

Example:
SQL is a programming language [1]. It performs CRUD operations [2].

Source:
*   [1] SQL notes.pdf, Page 3
*   [2] SQL notes.pdf, Page 5
"""
        elif has_web_sources and not has_document_sources:
            citation_instructions = """
**Citation Format for Web:**
- Write a comprehensive answer first
- At the end, list all source URLs under "Source:" with bullet points (*)

Example:
A Large Language Model (LLM) is a type of artificial intelligence...

Source:
*   https://example.com/article1
*   https://example.com/article2
"""
        else:
            citation_instructions = """
**Citation Format for Mixed Sources:**
- Use [1], [2] for document citations inline
- List web URLs at the end under "Source:" with bullet points (*)
"""
        
        return f"""You are a helpful AI assistant. Your task is to provide clear, well-structured answers.

**Chat History:**
{history_text if history_text else 'No previous conversation.'}

**User Question:** {query}

**Available Information:**
{sources}

**Instructions:** 
1. Provide a clear, comprehensive answer using the information above.
2. Write in a natural, informative style with proper paragraphs.
3. Structure your answer clearly with good flow.
{citation_instructions}

Write your response now:"""
    except Exception as e:
        print(f"Error in format_final_prompt: {e}")
        return f"Error formatting prompt: {str(e)}"


def has_rag_content(x):
    """Simple check if RAG has content (LLM validated)."""
    return isinstance(x, dict) and x.get("type") == "search_complete" and x.get("has_rag_content") == True


def no_rag_content(x):
    """Simple check if RAG has no content (LLM validated)."""
    return isinstance(x, dict) and x.get("type") == "search_complete" and x.get("has_rag_content") == False

# Create Conversational RAG Chain
def create_conversational_rag_chain(llm, vectorStore, appConfig, web_search_enabled=True):
    """
    Creates a Conversational RAG Chain using Runnable interface with RunnableBranch:
    - Uses RunnableBranch for conditional routing based on RAG results
    - RAG search first, then conditionally triggers web search
    - Handles meta-questions about conversation history
    """
    try:
       
        retriever = vectorStore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        
        rag_tool = create_rag_search_tool(retriever)
        web_tool = create_web_search_tool(appConfig, enabled=web_search_enabled)
        
       
        search_documents = create_search_documents_function(rag_tool, llm)
        add_web_search = create_add_web_search_function(web_tool, web_search_enabled)
        
        
        chain = (
            RunnableLambda(handle_meta_questions)
            | RunnableLambda(search_documents)
            | RunnableBranch(
               
                (has_rag_content, RunnableLambda(use_rag_only)),

                (no_rag_content, RunnableLambda(add_web_search)),
                RunnablePassthrough()
            )
            | RunnableLambda(format_final_prompt)
            | llm
            | StrOutputParser()
        )
        
        return chain
    except Exception as e:
        print(f"Error creating conversational RAG chain: {e}")
        import traceback
        traceback.print_exc()
        return None


def chat_loop(mongoDatabase, dbRecord, vectorStore, llm, appConfig, sessionId, web_search_enabled=None):
    """Main chat loop with MongoDB-backed memory using Runnable chain."""
    try:
        
        if web_search_enabled is None:
            print("\nWeb Search Option:")
            print("   [Y] Enable web search (search internet if document doesn't have answer)")
            print("   [N] Disable web search (only search documents)")
            web_choice = input("Enable web search? (Y/n): ").strip().lower()
            web_search_enabled = web_choice != 'n'
        
        # Create Runnable chain
        rag_chain = create_conversational_rag_chain(llm, vectorStore, appConfig, web_search_enabled)
        
        if rag_chain is None:
            print("Error: Failed to create RAG chain.")
            return
        
        # Get MongoDB chat history for this session
        chat_history = get_mongodb_chat_history(appConfig, sessionId)
        if chat_history is None:
            print("Error: Failed to connect to chat history.")
            return
        
        # Load existing history as messages
        history_messages = load_history_as_messages(chat_history)
        
        # Save session metadata
        save_session_metadata(mongoDatabase, sessionId, dbRecord["_id"], dbRecord["FolderName"])
        
        web_status = "Enabled" if web_search_enabled else "Disabled"
        print(f"\nChat Started | Folder: {dbRecord['FolderName']}")
        print(f"Session ID: {sessionId}")
        print(f"Web Search: {web_status}")
        print(f"Loaded {len(history_messages)} previous messages")
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
                    # Invoke Runnable chain with chat history
                    answer = rag_chain.invoke({
                        "input": userQuery,
                        "chat_history": history_messages
                    })
                    
                    # Clean the answer
                    answer = format_agent_response(answer)
                    
                    print(f"\nAssistant:\n{answer}")
                    
                    # Save to MongoDB chat history
                    chat_history.add_user_message(userQuery)
                    chat_history.add_ai_message(answer)
                    
                    # Update local history for next iteration
                    history_messages.append(HumanMessage(content=userQuery))
                    history_messages.append(AIMessage(content=answer))
                    
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

# User Interaction Menus

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
