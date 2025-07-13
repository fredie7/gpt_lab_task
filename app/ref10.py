# Necessary imports
import os
import pandas as pd
from dotenv import load_dotenv
from typing import Literal
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing.")
print("OPENAI_API_KEY found.")

# FastAPI app + CORS
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema
class SymptomInput(BaseModel):
    message: str

# Load and preprocess dataset once
def load_documents():
    if hasattr(load_documents, "cached"):
        return load_documents.cached

    print("📥 Loading and cleaning dataset...")
    df = pd.read_csv("symptoms_data.csv")
    df['symptom'] = df['symptom'].str.strip().str.lower()
    df['conditions'] = df['conditions'].apply(lambda x: [c.strip() for c in x.split(',')] if pd.notnull(x) else [])
    df['follow_up_questions'] = df['follow_up_questions'].apply(lambda x: [q.strip() for q in x.split(';')] if pd.notnull(x) else [])

    docs = [
        Document(
            page_content=f"symptom: {row['symptom']}\nfollow_up: {'; '.join(row['follow_up_questions'])}",
            metadata={
                "symptom": row['symptom'],
                "conditions": row['conditions'],
                "follow_up_questions": row['follow_up_questions']
            }
        )
        for _, row in df.iterrows()
    ]
    load_documents.cached = docs
    return docs

# Load documents
documents = load_documents()

# Embeddings
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# FAISS vectorstore caching
FAISS_INDEX_PATH = "faiss_index"
if os.path.exists(f"{FAISS_INDEX_PATH}/index.faiss"):
    print("Loading existing FAISS index...")
    vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
else:
    print("Creating FAISS index...")
    vectorstore = FAISS.from_documents(documents, embedding_model)
    vectorstore.save_local(FAISS_INDEX_PATH)

retriever = vectorstore.as_retriever(search_type="similarity", k=3)

# LLM setup
llm = ChatOpenAI(model="gpt-4o", temperature=0.2, openai_api_key=OPENAI_API_KEY)

# QA Chain
prompt = ChatPromptTemplate.from_messages([
    ("human", "Given the following context:\n\n{context}\n\nAnswer: {input}")
])
qa_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, qa_chain)

# Tools
@tool
def retrieve_diagnosis(symptom: str):
    """Return relevant diagnostic information based on symptom."""
    results = retriever.invoke(symptom)
    responses = "\n".join([doc.page_content for doc in results])
    return f"Follow-up questions for diagnosis:\n{responses}"

@tool
def give_recommendation(context: str) -> str:
    """Use the LLM to provide a medical recommendation based on the patient's symptoms."""
    prompt = (
        "Given the patient's symptom information below, provide a 2 line short, clear medical recommendation.\n\n"
        f"Symptom Context:\n{context}\n\n"
        "Recommendation:"
    )
    response = llm.invoke(prompt)
    return response.content.strip()

@tool
def explain_reasoning(context: str) -> str:
    """Use the LLM to explain the diagnostic reasoning based on the context."""
    prompt = (
        "Given the symptom context below, explain the likely diagnostic reasoning in simple terms.\n\n"
        f"Symptom Context:\n{context}\n\n"
        "Diagnostic Reasoning:"
    )
    response = llm.invoke(prompt)
    return response.content.strip()

# Precreate agents
diagnostic_agent = create_react_agent(llm, tools=[retrieve_diagnosis], prompt="You are a diagnostician.")
recommendation_agent = create_react_agent(llm, tools=[give_recommendation], prompt="You are a medical advisor.")
explanation_agent = create_react_agent(llm, tools=[explain_reasoning], prompt="You explain reasoning behind diagnoses.")

# LangGraph workflow
members = ["diagnostic", "recommendation", "explanation"]
class Router(TypedDict):
    next: Literal["diagnostic", "recommendation", "explanation", "FINISH"]

class State(MessagesState):
    next: str

system_prompt = f"""
You are a controller managing the workflow: {members}.
Start with 'diagnostic', then 'recommendation', and finish with 'explanation'.
Once complete, choose FINISH.
"""

def supervisor_node(state: State) -> Command[Literal["diagnostic", "recommendation", "explanation", "__end__"]]:
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = llm.with_structured_output(Router).invoke(messages)
    next_step = response["next"]
    return Command(goto=END if next_step == "FINISH" else next_step, update={"next": next_step})

def diagnostic_node(state: State) -> Command[Literal["supervisor"]]:
    result = diagnostic_agent.invoke(state)
    return Command(update={"messages": [HumanMessage(content=result['messages'][-1].content, name="diagnostic")]}, goto="supervisor")

def recommendation_node(state: State) -> Command[Literal["supervisor"]]:
    result = recommendation_agent.invoke(state)
    return Command(update={"messages": [HumanMessage(content=result['messages'][-1].content, name="recommendation")]}, goto="supervisor")

def explanation_node(state: State) -> Command[Literal["supervisor"]]:
    result = explanation_agent.invoke(state)
    return Command(update={"messages": [HumanMessage(content=result['messages'][-1].content, name="explanation")]}, goto="supervisor")

# Build and compile graph
graph = StateGraph(State)
graph.add_node("supervisor", supervisor_node)
graph.add_node("diagnostic", diagnostic_node)
graph.add_node("recommendation", recommendation_node)
graph.add_node("explanation", explanation_node)
graph.set_entry_point("supervisor")
graph.add_edge(START, "supervisor")
result_graph = graph.compile()

# if __name__ == "__main__":
#     test_result = result_graph.invoke({"messages": [("user", "I feel fatigued.")]})
#     print("\nFinal output:\n", test_result['messages'][-1].content)

# Endpoint for symptom message
@app.post("/ask")
async def diagnose(input_data: SymptomInput):
    user_message = input_data.message
    state_input = {"messages": [("user", user_message)]}
    result = result_graph.invoke(state_input)
    answer = result["messages"][-1].content
    return {"response": answer}

# Test run (can be removed later)

