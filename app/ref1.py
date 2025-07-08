from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.docstore.document import Document

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not found.")

# FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class ChatRequest(BaseModel):
    question: str

# Extract and preprocess documents
def extract_csv_info(file_path):
    dataset = pd.read_csv(file_path)
    dataset['combined_text'] = (
        dataset['symptom'].fillna('') + ' ' +
        dataset['conditions'].fillna('') + ' ' +
        dataset['follow_up_questions'].fillna('')
    )
    documents = [Document(page_content=text) for text in dataset['combined_text'].tolist()]
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
    return splitter.split_documents(documents)

# Create vector store
def create_vector_store(documents):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(documents, embedding=embeddings)

# Define multi-agent retrieval chain
def create_agent_chain(vector_store):
    model = ChatOpenAI(model="gpt-4o", temperature=0.1)

    base_prompt = ChatPromptTemplate.from_messages([
        ("system", """
            You are a collaborative AI health assistant composed of three agents:

            1. Diagnostic Agent - Asks follow-up questions based on user symptoms.
            2. Recommendation Agent - Provides advice based on symptom patterns and likely conditions.
            3. Explanation Agent - Explains the logic behind any diagnosis or recommendation in a simple and empathetic way.

            Use this context:
            {context}

            Follow the steps:
            - Ask follow-up questions to clarify symptoms (Diagnostic Agent)
            - Provide suggestions based on input and conditions (Recommendation Agent)
            - Add a rationale behind your output (Explanation Agent)
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    chain = create_stuff_documents_chain(llm=model, prompt=base_prompt)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm=model,
        retriever=retriever,
        prompt=retriever_prompt
    )

    return create_retrieval_chain(history_aware_retriever, chain)

# Initialize
csv_file_path = "symptoms_data.csv"
documents = extract_csv_info(csv_file_path)
vector_store = create_vector_store(documents)
retrieval_chain = create_agent_chain(vector_store)
chat_history = []

# Endpoint
@app.post("/ask")
async def chat(request: ChatRequest):
    global chat_history
    try:
        result = retrieval_chain.invoke({
            "chat_history": chat_history,
            "input": request.question
        })
        answer = result["answer"]
        chat_history.append(HumanMessage(content=request.question))
        chat_history.append(AIMessage(content=answer))
        return {"response": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
