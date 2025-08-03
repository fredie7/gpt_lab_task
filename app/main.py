# Import dependencies

# To run blocking code asyncronously
import asyncio
# To Create web API
from fastapi import FastAPI
# To allow for communication between the client side and the server
from fastapi.middleware.cors import CORSMiddleware

# To acknowledge human message in the conversation
from langchain_core.messages import HumanMessage
# Import the custom agent logic and session store
from agents import run_agent_loop, conversation_store
from models import SymptomInput


# Initialize FastAPI
app = FastAPI()

# Add CORS middleware to facilitate communication between the client and server
app.add_middleware(
    CORSMiddleware,
    # Allow any origin (*) or website and localhost client
    allow_origins=["*", "http://localhost:3000/"],
    allow_credentials=True,
    # Allow all HTTP methods
    allow_methods=["*"],
    # Allow all headers
    allow_headers=["*"],
)

# Define the first version of the endpoint to handle user requests
@app.post("/api/v1/ask")

# Handle the request body containing the user's message, history, and session_id.
async def ask(input_data: SymptomInput):

    # Clean up the user's message
    user_msg = input_data.message.strip()

    # Ensure session ID is stripped of whitespace
    session_id = input_data.session_id.strip()

    # Initialize message list if session doesn't exist
    if session_id not in conversation_store:
        conversation_store[session_id] = []

    # Add user message to a local copy of the conversation history for the current session
    local_messages = conversation_store[session_id] + [HumanMessage(content=user_msg)]

    # Wrap the messages into a state object, which is required by the LangGraph agent
    state = {"messages": local_messages}

    # Run the medical agent in separate thread to avoid blocking the server due to time 
    # spent in the interaction between it and it's co-workers(tools)
    state = await asyncio.to_thread(run_agent_loop, state, session_id)

    # Save the updated conversation back into the global session store
    conversation_store[session_id] = state["messages"]

    # Get the last message(medical agent) from the state to return as the response
    last_msg = state["messages"][-1]
  
    # Return the content of the last message as the response
    return {"response": last_msg.content}
