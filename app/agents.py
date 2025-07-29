from typing import Sequence, Annotated, Dict
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from tools import tools
from data_loader import llm

# Bind the tools to the LLM
llm = llm.bind_tools(tools)

# Define the state that manages information between users and all agents
class MedicalAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Supervisory agent function
def medical_agent(state: MedicalAgentState) -> MedicalAgentState:
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
    print("[Agent] checking tools for diagnosis, recommendation or explanation...")
    response = llm.invoke([system_prompt] + state['messages'])

    print("Checking for tool calls...")
    for i, tool_call in enumerate(response.tool_calls):
        print(f" Tool #{i + 1}: {tool_call.get('name', 'UnknownTool')}")
        print(f" Args: {tool_call.get('args', {})}")

    return {"messages": [response]}

# Tool control
def should_continue(state: MedicalAgentState):
    messages = state["messages"]
    last_message = messages[-1]
    return "continue" if last_message.tool_calls else "end"

# State graph
graph = StateGraph(MedicalAgentState)
graph.add_node("medical_agent", medical_agent)
graph.add_node("tools", ToolNode(tools=tools))
graph.set_entry_point("medical_agent")
graph.add_conditional_edges("medical_agent", should_continue, {
    "continue": "tools",
    "end": END
})
graph.add_edge("tools", "medical_agent")

# Compile agent app
agent_app = graph.compile()

# Conversation store for sessions
conversation_store: Dict[str, list[BaseMessage]] = {}

# Agent loop runner
def run_agent_loop(state, session_id):
    local_messages = state["messages"].copy()
    while True:
        state = agent_app.invoke({"messages": local_messages})
        last_msg = state["messages"][-1]
        local_messages.append(last_msg)
        if not getattr(last_msg, "tool_calls", None):
            break
    return {"messages": local_messages}
