# To facilitate necessary typing and data structure utilities
from typing import Sequence, Annotated, Dict
# To allow for the definiition of dict with typed fields
from typing_extensions import TypedDict

# Import message and agent graph components from LangChain and LangGraph
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

# Import predefined tool
from tools import tools
from data_loader import llm

# Bind the tools to the LLM to allow it call the appropriate tool during conversation
llm = llm.bind_tools(tools)

# Define the state that manages information between users and all agents.
class MedicalAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Supervisory medical agent function
def medical_agent(state: MedicalAgentState) -> MedicalAgentState:
    # Set the behaviour of the medical agent
    system_prompt = SystemMessage(
        content=f"""
            You are a medical assistant responsible for managing the conversation among these worker agents: {tools}.
            - First ask 'Hello! Before we proceed, could you please provide your name, age, and gender? This is to help me get to know you'.
            -Include the patient's name in the conversation to make it personalized, but not on every response
            - For each response, start with the corresponding agent or tool responsible for instance (Diagnostic Agent):, (Recommendation Agent):, (Explanation Agent):. Include the brackets.
            - Provide diagnostic questions to examine the patient
            - Though you would discover more than one question from the diagnostic agent, ask them one at a time.
            - Even if the patient decides to respond with short yes, no or short vague answers, convey the entire context to the recommender agent.
            - Don't include any technical error messages in your responses
            - After providing a recommendation, ask the user if they need an explanation for the recommendation.
        """
    )
    # Start tracking of tools
    print("[Agent] checking tools for diagnosis, recommendation or explanation...")
    # Make llm acknowledge it's behaviour from the system prompt as well as previous conversation history
    response = llm.invoke([system_prompt] + state['messages'])

    # Check for the tools that are being called
    print("Checking for tool calls...")
    for i, tool_call in enumerate(response.tool_calls):
        print(f" Tool #{i + 1}: {tool_call.get('name', 'UnknownTool')}")
        print(f" Args: {tool_call.get('args', {})}")
    # Return updated messages including response from the medical agent
    return {"messages": [response]}

# Help the medical agent determine if any tool is still being called - Otherwise, have it continue
def should_continue(state: MedicalAgentState):
    # Collect all messages in the state
    messages = state["messages"]
    # Track the last message
    last_message = messages[-1]
    return "continue" if last_message.tool_calls else "end"

# Build the connecting graph between the medical agent and its co-workers(tools)
graph = StateGraph(MedicalAgentState)
# Add medical agent's node to the graph
graph.add_node("medical_agent", medical_agent)
# Add the tool node to handle the execution of the appropriate tool
graph.add_node("tools", ToolNode(tools=tools))
# Set entry point here the conversation starts
graph.set_entry_point("medical_agent")
# Define conditional transitions from "medical_agent"
graph.add_conditional_edges("medical_agent", should_continue, {
    "continue": "tools",
    "end": END
})
# Find the medical agent once tools are executed to deliver appropriate response to the user
graph.add_edge("tools", "medical_agent")

# Compile the graph into an agent
agent_app = graph.compile()

# Create conversation store of session_id from each user
conversation_store: Dict[str, list[BaseMessage]] = {}

# Agent loop to run the conversation
def run_agent_loop(state, session_id):
    # Copy current messages locally to avoid modifying the global store directly
    local_messages = state["messages"].copy()
    # Loop through until no more tool calls are detected
    while True:
        # Invoke the graph
        state = agent_app.invoke({"messages": local_messages})
        # Get the last message
        last_msg = state["messages"][-1]
        # Append the last response
        local_messages.append(last_msg)
        # If no tool was called, exit loop
        if not getattr(last_msg, "tool_calls", None):
            break
    # Return the conversation    
    return {"messages": local_messages}
