# Import dependencies
import os
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Literal, Dict,TypedDict, Sequence, Annotated
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import RetrievalQA
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langgraph.graph import StateGraph,START,END,MessagesState
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage,ToolMessage,BaseMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from IPython.display import display, Image
from pprint import pprint


from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# handle multiple users at a time
import asyncio

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing.")
print("OPENAI_API_KEY found.")

# Initialize FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*","http://localhost:3000/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the server's schema
class SymptomInput(BaseModel):
    message: str
    history: list[dict] = [] 
    session_id: str

# Load and preprocess dataset once
def load_documents():
    if hasattr(load_documents, "cached"):
        return load_documents.cached

    print("Loading and cleaning dataset...")
    df = pd.read_csv("symptoms_data.csv")
    dataset_columns = ['symptom', 'conditions', 'follow_up_questions']

    for col in dataset_columns:
        if df[col].isnull().any():
            print(f"The column '{col}' has null values.")
        else:
            print(f"The column '{col}' does not have a null value.")

    df['symptom'] = df['symptom'].str.strip().str.lower()
    df['conditions'] = df['conditions'].apply(lambda x: [c.strip().lower() for c in x.split(',')])
    df['follow_up_questions'] = df['follow_up_questions'].apply(lambda x: [q.strip().lower() for q in x.split(';')])

    docs = [
        Document(
            page_content=f"symptom: {row['symptom']}\nconditions: {', '.join(row['conditions'])}\nfollow_up: {'; '.join(row['follow_up_questions'])}",
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

# Perform RAG(Retrieval-Augmented Generation)

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)

# Split documents into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=500
)

# Split documents into smaller chunks for better retrieval to help the model retrieve relevant context
documents_split = text_splitter.split_documents( documents)

# Create a vector store using FAISS to store the  documents within specified index
vectorstore = FAISS.from_documents(documents, embeddings)

# Initialize the LLM and set the temperature in a way to prevent hallucination
llm = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY,temperature=0.0)

# Create a retrieval chain to handle the retrieval of relevant documents and generate responses based on user input
chain = RetrievalQA.from_chain_type(llm=llm, chain_type="map_reduce", retriever=vectorstore.as_retriever())
# print("CHAIN===>>", chain.invoke("fatigue"))

# Define tools decorator for the agent to use
@tool
# Create a diagnostic agent that provides follow-up diagnostic questions based on the symptom to help another medical assistant make recommendations.
def provide_diagnosis(symptom: str) -> str:
    """Return follow-up diagnostic questions based on the symptom to help another medical assistant make recommendations."""

    # retrieve top 1 match to the user's symptom
    docs = vectorstore.similarity_search(symptom, k=3)  

    # Initialize an empty list to store follow-up questions
    follow_up_questions = [] 

    # Iterate through the retrieved documents to extract follow-up questions
    for doc in docs:  
        # Get only follow-up questions from the raft of metadata, or nothing if it does not exist
        questions = doc.metadata.get("follow_up_questions", [])  
        if questions:
            # Add retrieved questions to the list
            follow_up_questions.extend(questions)  

    if follow_up_questions:
        # Return unique questions to avoid repetition
        return "I have some questions to help your diagnosis:\n- " + "\n- ".join(set(follow_up_questions)) 
    else:
        return f"I have no knowledge about the symptom: '{symptom}'. Please tell me about a closely related symptom to help us discuss how you feel."

@tool
# Create a recommendation agent that provides medical recommendations based on the user's symptom and diagnostic agent's questions.
def provide_recommendation(context: str) -> str:
    """
    Generate a short, friendly, and clear medical recommendation based strictly on the patient's symptom input.
    The output must include the symptom keyword and avoid telling the patient to visit a doctor.
    The recommendation should be grounded in the dataset and limited to no more than three sentences.
    """
    # Use the context to generate a recommendation from patient's diagnosis
    prompt = (
        "You are a helpful medical assistant using only the provided dataset to make recommendations.\n"
        "Based on the symptom described below, generate a short, user-friendly recommendation.\n"
        "Do NOT suggest visiting a doctor. Do NOT invent medical facts.\n"
        f"Make sure to include the keyword: '{context.strip()}' so another assistant can explain your reasoning.\n"
        "Keep your answer to a maximum of three sentences.\n\n"
        f"Symptom: {context.strip()}\n\n"
        "Recommendation:"
    )
    # Invoke the retrieval chain to get relevant recommendation based on the context
    response = chain.invoke(prompt)
    # Return the content of the response
    return response.content.strip()

@tool
# Create an explanation agent that explains the reasoning behind a recommendation provided by the recommender agent.
def provide_explanation(context: str) -> str:
    """
    Explain in simple terms the reasoning behind a recommendation previously given,
    using only knowledge inferred from the dataset.
    The output should be user-friendly, factually grounded, and avoid speculation.
    """
    # Use the context to generate an explanation based on the provided information by the recommendation agent
    prompt = (
        "You are a healthcare assistant who explains recommendations in simple, clear terms.\n"
        "Given the context of a previous recommendation, explain the most likely reasons behind it.\n"
        "Use only knowledge from the dataset and do not speculate or invent causes.\n\n"
        f"Recommendation Context:\n{context.strip()}\n\n"
        "Explanation:"
    )
    # Invoke the retrieval chain to include relevant explanation based on the context
    response = chain.invoke(prompt)
    # Return the content of the response
    return response.content.strip()

# List of tools for the supervisory agent to use since it is responsible for managing the conversation among the worker agents in a "ReAct" pattern.
tools = [provide_diagnosis,provide_recommendation,provide_explanation]

# Bind the tools to the LLM to aid contextual interaction with the supervisory agent
llm = llm.bind_tools(tools) 

# Define the state that manages information between users and all agents
class MedicalAgentState(TypedDict):
  # Create a list of messages in the conversation, including user inputs and agent responses.
  messages: Annotated[Sequence[BaseMessage],add_messages] 

# Define the supervisory agent that manages the conversation among the worker agents as well as the user.
def medical_agent(state: MedicalAgentState) -> MedicalAgentState:

    # Set the system prompt to guide the agent's behavior
    system_prompt = SystemMessage(
        content=f"""
            You are a medical assistant responsible for managing the conversation among these worker agents: {tools}.
            - First ask 'Hello! Before we proceed, could you please provide your name, age, and gender? This is to help me get to know you'.
            -Include the patient's name in the conversation to make it personalized, but not on every response
            - For each response, start with the corresponding agent or tool responsible for instance (Diagnostic Agent):, (Recommendation Agent):, (Explanation Agent):.
            - Provide diagnostic questions to examine the patient
            - Though you would discover more than one question from the diagnostic agent, ask them one at a time.
            - Even if the patient decides to respond with short yes, no or short vague answers, convey the entire context to the recommender agent.
            - Don't include any technical error messages in your responses
            - After providing a recommendation, ask the user if they need an explanation for the recommendation.
        """
    )

    # Track the tool calls made by the supervisory agent to other agents
    print("[Agent] checking tools for diagnosis, recommendation or explanation...")
    
    # Invoke the LLM with the system prompt and the current messages in the state
    response = llm.invoke([system_prompt] + state['messages'])

    # Track Tools
    print("Checking for tool calls...")
    for i, tool_call in enumerate(response.tool_calls):
        # Print the tool name
        print(f" Tool #{i + 1}: {tool_call.get('name', 'UnknownTool')}")
        # Print the arguments passed to the tool
        print(f" Args: {tool_call.get('args', {})}")
   
    # Add the response to the messages in the state
    return {"messages": [response]}

# Check if the last message has tool calls
# If it does, continue the conversation; otherwise, end it.
def should_continue(state: MedicalAgentState):
    # Get the list of messages in the conversation
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls: 
        return "end"
    else:
        return "continue"

# Create a state graph to manage the conversation flow
graph = StateGraph(MedicalAgentState)

# Add the supervisory agent node to the graph
graph.add_node("medical_agent",medical_agent)

# Create a node for the agentic tools(diagnostic,recommender & explanatory) to be invoked by the supervisory agent when needed
tool_node = ToolNode(tools=tools)

# Include the tool node in the graph
graph.add_node("tools",tool_node)

# Set the entry point of the graph to the supervisory agent
# This is where the conversation starts
graph.set_entry_point("medical_agent")

# Add edges to the graph to control the flow of conversation
graph.add_conditional_edges(
    "medical_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)

# Add an edge from the tool node back to the supervisory agent to continue the conversation where necessary
graph.add_edge("tools", "medical_agent")

# Compile the graph to create the agent application
agent_app = graph.compile()

# Create conversation store to handle multiple users
conversation_store: Dict[str, list[BaseMessage]] = {}

# Define the endpoint to handle user requests
@app.post("/ask")

# Handle the request body containing the user's message, history, and session_i
async def ask(input_data: SymptomInput):

    # Extract user message and session ID from the input data
    user_msg = input_data.message.strip()

    # Ensure session ID is stripped of whitespace
    session_id = input_data.session_id.strip()

    # Initialize session if it doesn't exist
    if session_id not in conversation_store:
        conversation_store[session_id] = []

    # Add user message to a local copy of the conversation history for the current session
    local_messages = conversation_store[session_id] + [HumanMessage(content=user_msg)]

    # Create the state for the agent application with the local messages for managing conversation flow
    state = {"messages": local_messages}

    # Run blocking agent in separate thread
    state = await asyncio.to_thread(run_agent_loop, state, session_id)

    # Update session store
    conversation_store[session_id] = state["messages"]

    # Get the last message from the state to return as the response
    last_msg = state["messages"][-1]
  
    # Return the content of the last message as the response
    return {"response": last_msg.content}



# Define the function to run the agent loop
def run_agent_loop(state, session_id):

    # Create a local copy of the messages to avoid mutating the original state
    local_messages = state["messages"].copy()

    while True:
        # Invoke the agent application with the current messages
        state = agent_app.invoke({"messages": local_messages})

        # Get the last message from the state
        last_msg = state["messages"][-1]

        # Append the last message to the local copy of messages
        local_messages.append(last_msg)

        # Check if the last message contains any tool calls
        if not getattr(last_msg, "tool_calls", None):
            # If there are no tool calls, end the loop
            break
    # Return the final state
    return {"messages": local_messages}




