# Import dependencies
import os
import asyncio
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from typing import Dict
from agents import agent_app, run_agent_loop, conversation_store
from data_loader import load_documents

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
    allow_origins=["*", "http://localhost:3000/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the server's schema
class SymptomInput(BaseModel):
    message: str
    history: list[dict] = []
    session_id: str

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
