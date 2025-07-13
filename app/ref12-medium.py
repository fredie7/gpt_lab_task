# ✅ Necessary imports
import os
import pandas as pd
from dotenv import load_dotenv
from typing import Literal
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.utils.function_calling import convert_to_openai_function
from typing import TypedDict, Annotated, Sequence
import operator
# from langgraph.prebuilt import ToolExecutor

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ✅ Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY is missing.")
print("✅ OPENAI_API_KEY found.")

# ✅ FastAPI app + CORS
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Input schema
class SymptomInput(BaseModel):
    message: str

# ✅ Load and preprocess dataset once
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

# ✅ Load documents
documents = load_documents()

# ✅ Embeddings
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# ✅ FAISS vectorstore caching
FAISS_INDEX_PATH = "faiss_index"
if os.path.exists(f"{FAISS_INDEX_PATH}/index.faiss"):
    print("✅ Loading existing FAISS index...")
    vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
else:
    print("📦 Creating FAISS index...")
    vectorstore = FAISS.from_documents(documents, embedding_model)
    vectorstore.save_local(FAISS_INDEX_PATH)

retriever = vectorstore.as_retriever(search_type="similarity", k=3)

# ✅ LLM setup
llm = ChatOpenAI(model="gpt-4o", temperature=0.2, openai_api_key=OPENAI_API_KEY)

# ✅ QA Chain
prompt = ChatPromptTemplate.from_messages([
    ("human", "Given the following context:\n\n{context}\n\nAnswer: {input}")
])
qa_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, qa_chain)

# ✅ Tools
def create_agent(llm, tools, system_message:str):
    """Create an agent with specified LLM, tools, and system prompt."""
    functions = [convert_to_openai_function(tool) for tool in tools]
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="messages")
    ])
    return prompt | llm.bind_functions(functions)

def get_diagnostics(symptom: str) -> str:
    """Retrieve diagnosis based on user symptom."""
    result = retrieval_chain.invoke(symptom)
    return result
get_diagnostics_tools = [get_diagnostics]
diagnostic = create_agent(llm, get_diagnostics_tools, "You are a medical diagnosis assistant. Use tools to fetch questions regarding the symptoms.")

def make_recommendations(diagnosis: str) -> str:
    """Retrieve recommendations based on user diagnosis."""
    result = retrieval_chain.invoke(diagnosis)
    return result
make_recommendations_tools = [make_recommendations]
recommender = create_agent(llm, make_recommendations_tools, "You are a medical recommendation assistant. Use tools to make recommendations regarding the symptoms from the answers provided by the user during diagnosis.")

def explain_recommendations(diagnosis: str) -> str:
    """Retrieve explanations for recommendations based on user diagnosis."""
    result = retrieval_chain.invoke(diagnosis)
    return result
get_diagnostics_tools = [explain_recommendations]
explainer = create_agent(llm, get_diagnostics_tools, "You are a medical assistant. Use tools to explain the recommendations based on the user's diagnosis.")


class MultiAgentState(TypedDict):
    """State for the multi-agent workflow."""
    messages: Annotated[Sequence[HumanMessage], operator.add]
    last_active_agent: str

# tool_executor = ToolExecutor(tools=[get_diagnostics, make_recommendations, explain_recommendations])

# Supervisor Agent
supervisor_prompt = """
You are a supervisor coordinating medical diagnosis, recommendation and explanation tasks. Based on the user input, route to:
- 'diagnostic' for medical diagnosis
- 'recommendation' for treatment recommendations
- 'explanation' for explanations of recommendations
Return the name of the agent to handle the task.
"""
supervisor = create_agent(llm, [], supervisor_prompt)

def supervisor_node(state):
    result = supervisor.invoke(state)
    return {"last_active_agent": result["messages"][-1].content}

# Agent Nodes
def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [AIMessage(content=result["messages"][-1].content, name=name)]}

# Define Graph
graph = StateGraph(MultiAgentState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("diagnostic", lambda state: agent_node(state, diagnostic, "diagnostic"))
graph.add_node("recommender", lambda state: agent_node(state, recommender, "recommender"))
graph.add_node("explainer", lambda state: agent_node(state, explainer, "explainer"))

graph.add_edge(START,"supervisor")
graph.add_conditional_edges(
    "supervisor",
    lambda state: state["last_active_agent"],
    {
        "diagnostic": "diagnostic",
        "recommender": "recommender",
        "explainer": "explainer",
        "end": END
    }
)
graph.add_edge("diagnostic", "supervisor")
graph.add_edge("recommender", "supervisor")
graph.add_edge("explainer", "supervisor")

retrieved_graph = graph.compile()

user_input = "I have a headache"
state = {"messages": [HumanMessage(content=user_input)], "last_active_agent": ""}

# Run the graph
result = retrieved_graph.invoke(state)

# Print results
for message in result["messages"]:
    print(f"{message.name or 'User'}: {message.content}")
