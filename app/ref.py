from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import pandas as pd

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import LLMChain
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains import create_retrieval_chain

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is missing.")

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema with personalization
class ChatRequest(BaseModel):
    question: str
    age: Optional[int] = None
    gender: Optional[str] = None
    known_conditions: Optional[List[str]] = []

# Load and process documents
def load_documents(csv_file: str):
    df = pd.read_csv(csv_file)
    df['combined_text'] = (
        df['symptom'].fillna('') + ' ' +
        df['conditions'].fillna('') + ' ' +
        df['follow_up_questions'].fillna('')
    )
    documents = [Document(page_content=row) for row in df['combined_text']]
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
    return splitter.split_documents(documents)

# Build vector store
def build_vectorstore(docs):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(docs, embedding=embeddings)

# Create individual agent chains
def create_agent_chain(system_prompt):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    return LLMChain(llm=ChatOpenAI(model="gpt-4o", temperature=0.1), prompt=prompt)

# Create all three agents
system_prompts = {
    "diagnostic": "You are a diagnostic agent. Ask targeted questions based on symptoms to identify possible conditions.",
    "recommendation": "You are a recommendation agent. Suggest lifestyle or medical advice based on diagnosis and user profile.",
    "explanation": "You are an explanation agent. Clearly and empathetically explain the reasoning behind the diagnosis and recommendation."
}

diagnostic_chain = create_agent_chain(system_prompts["diagnostic"])
recommendation_chain = create_agent_chain(system_prompts["recommendation"])
explanation_chain = create_agent_chain(system_prompts["explanation"])

# Load docs and create vector store
csv_path = "symptoms_data.csv"
documents = load_documents(csv_path)
vectorstore = build_vectorstore(documents)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Enable history-aware retrieval
history_retriever_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])

history_aware_retriever = create_history_aware_retriever(
    llm=ChatOpenAI(model="gpt-4o"),
    retriever=retriever,
    prompt=history_retriever_prompt
)

chat_history = []

# Multi-agent controller
async def multi_agent_pipeline(question, chat_history, profile):
    # Agent 1: Diagnostic
    diagnosis = diagnostic_chain.invoke({
        "input": question,
        "chat_history": chat_history,
        **profile
    })
    chat_history.append(AIMessage(content=diagnosis["text"]))

    # Agent 2: Recommendation
    recommendation = recommendation_chain.invoke({
        "input": diagnosis["text"],
        "chat_history": chat_history,
        **profile
    })
    chat_history.append(AIMessage(content=recommendation["text"]))

    # Agent 3: Explanation
    explanation = explanation_chain.invoke({
        "input": recommendation["text"],
        "chat_history": chat_history,
        **profile
    })
    chat_history.append(AIMessage(content=explanation["text"]))

    return {
        "diagnosis": diagnosis["text"],
        "recommendation": recommendation["text"],
        "explanation": explanation["text"]
    }

@app.post("/ask")
async def ask_question(req: ChatRequest):
    global chat_history
    chat_history.append(HumanMessage(content=req.question))

    profile = {
        "age": req.age or "Unknown",
        "gender": req.gender or "Unknown",
        "known_conditions": ", ".join(req.known_conditions) if req.known_conditions else "None"
    }

    try:
        response = await multi_agent_pipeline(req.question, chat_history, profile)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
