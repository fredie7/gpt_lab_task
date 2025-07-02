from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import os
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.docstore.document import Document  # Correct import for Document class

from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not not found.")
else:
    print("OPENAI_API_KEY is loaded successfully.")


# Define the FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# Define the request model
class ChatRequest(BaseModel):
    question: str

# Perform RAG
# Information retrieval from the CSV file using the specified columns
def extract_csv_info(file_path):
    dataset = pd.read_csv(file_path)
    
    # Concatenate relevant columns into a single text field for each row
    dataset['combined_text'] = (
        dataset['symptom'].fillna('') + ' ' +
        dataset['conditions'].fillna('') + ' ' +
        dataset['follow_up_questions'].fillna('')
    )
    
    # Convert each row into a Document object
    documents = [Document(page_content=text) for text in dataset['combined_text'].tolist()]
    
    split_docs = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=20
    )
    splitDocs = split_docs.split_documents(documents)
    return splitDocs

# Create the vector store from documents
def create_vector_store(documents):
    embedding = OpenAIEmbeddings()
    vectorStore = FAISS.from_documents(documents, embedding=embedding)
    return vectorStore

def create_recurring_chain(vectorStore):
    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0.1
    )
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
        You are a collaborative AI health assistant composed of three agents:

        1. **Diagnostic Agent** - Asks follow-up questions based on user symptoms.
        2. **Recommendation Agent** - Provides advice based on symptom patterns and likely conditions.
        3. **Explanation Agent** - Explains the logic behind any diagnosis or recommendation in a simple and empathetic way.

        Your data includes:
        - "symptom": user's complaint
        - "conditions": possible diagnoses
        - "follow_up_questions": dynamic questions to narrow down conditions

        Instructions:
        - When a user provides a symptom, start with a question (Diagnostic Agent).
        - Ask follow-up questions until enough information is gathered.
        - Then respond with advice (Recommendation Agent).
        - Follow that with an explanation of the reasoning (Explanation Agent).
        - Stay in character and be concise, caring, and medically responsible.

        Relevant data:
        {context}
            """
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    # 🌐 LLM chain that incorporates documents into the multi-agent prompt
    chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )

    # 🔍 Retriever setup
    retriever = vectorStore.as_retriever(search_kwargs={"k": 3})

    # 🧠 Retriever prompt (for history awareness)
    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    #  Make the retriever aware of chat history
    history_aware_retriever = create_history_aware_retriever(
        llm=model,
        retriever=retriever,
        prompt=retriever_prompt
    )

    # 🔗 Final chain: combines history-aware retriever with the agent prompt
    retrieval_chain = create_retrieval_chain(
        history_aware_retriever,
        chain
    )

    return retrieval_chain


# Initialize the documents and chain globally
csv_file_path = "symptoms_data.csv"
documents = extract_csv_info(csv_file_path)
vectorStore = create_vector_store(documents)
chain = create_recurring_chain(vectorStore)

# Initialize chat history
chat_history = []

@app.post("/ask")
async def chat(request: ChatRequest):
    global chat_history
    question = request.question
    
    # Process the chat and return the response
    try:
        response = chain.invoke({
            "chat_history": chat_history,
            "input": question, 
        })
        answer = response["answer"]
        
        # Update chat history
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))
        
        return {"response": answer}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)