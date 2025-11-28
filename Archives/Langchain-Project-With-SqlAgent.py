import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

# 1. Configuration & Setup
def load_app_configuration():
    """Loads environment variables."""
    load_dotenv()
    googleApiKey = os.getenv("GOOGLE_API_KEY")

    if not googleApiKey:
        print("Error: GOOGLE_API_KEY not found in .env file.")
        exit()

    return {"googleApiKey": googleApiKey}

def get_gemini_llm(apiKey):
    """Initializes the Gemini LLM."""
    try:
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            google_api_key=apiKey,
            temperature=0
        )
    except Exception as e:
        print(f"Error initializing Gemini: {e}")
        return None

# 2. CSV to SQL Logic
def create_db_from_csv(csvPath):
    """
    Reads a CSV and converts it to a temporary in-memory SQLite database.
    """
    try:
        if not os.path.exists(csvPath):
            print(f"Error: File not found at {csvPath}")
            return None

        fileName = os.path.basename(csvPath).split('.')[0]
        dataFrame = pd.read_csv(csvPath)
        
        engine = create_engine("sqlite:///:memory:")
        
        dataFrame.to_sql(fileName, engine, index=False)
        
        print(f"Successfully loaded '{fileName}' into temporary SQL database.")
        print(f" - Rows: {len(dataFrame)}")
        print(f" - Columns: {list(dataFrame.columns)}")
        
        return SQLDatabase(engine=engine)
        
    except Exception as e:
        print(f"Error converting CSV to SQL: {e}")
        return None

# 3. Agent Setup
def initialize_sql_agent(database, llm):
    """Creates the LangChain SQL Agent."""
    try:
        toolkit = SQLDatabaseToolkit(db=database, llm=llm)
        
        agentExecutor = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True, 
            agent_type="zero-shot-react-description",
            handle_parsing_errors=True
        )
        return agentExecutor
    except Exception as e:
        print(f"Error creating SQL Agent: {e}")
        return None

# 4. Interaction Loop
def chat_loop(agentExecutor):
    """Main user input loop."""
    print("\n💬 SQL Agent Active (Type 'exit' to stop, 'new' to change file)\n")
    
    while True:                      
        userQuery = input("\nAsk a question about your data: ").strip()
        
        if userQuery.lower() == 'exit':
            return "exit"
        if userQuery.lower() == 'new':
            return "new"

        print("Analyzing data...", end="\r")

        try:
            response = agentExecutor.invoke({"input": userQuery})
            
            print("\n✨ Answer:")
            print(response["output"])
            print("-" * 50)
            
            
            inputTokensEst = len(userQuery) / 4 
            outputTokensEst = len(response["output"]) / 4

            print(f"Estimated Tokens: ~{int(inputTokensEst + outputTokensEst)}")
            
        except Exception as e:
            print(f"\nError generating response: {e}")

# 5. Main Execution
def main():
    appConfig = load_app_configuration()
    
    print("Initializing Gemini Model...")
    llm = get_gemini_llm(appConfig["googleApiKey"])
    
    while True:
        print("\n" + "=" * 50)
        print("Gemini CSV-to-SQL Data Analyst")
        print("=" * 50)
        
        csvPath = input("Enter the path to your CSV file (e.g., data.csv): ").strip()
        
      
        csvPath = csvPath.replace('"', '').replace("'", "")
        
        database = create_db_from_csv(csvPath)
        
        if database:
            agentExecutor = initialize_sql_agent(database, llm)
            
            if agentExecutor:
                status = chat_loop(agentExecutor)
                
                if status == "exit":
                    print("👋 Goodbye!")
                    break
            else:
                print("Failed to initialize agent.")

if __name__ == "__main__":
    main()