# Necessary imports
import os
import pandas as pd
from dotenv import load_dotenv
from typing import Literal, Dict,TypedDict, Sequence, Annotated
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
    history: list[dict] = []  # Expecting [{"role": "user"/"assistant", "content": "..."}]


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

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
documents_split = text_splitter.split_documents( documents)
vectorstore = FAISS.from_documents(documents, embeddings)
llm = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY,temperature=0.0)
chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
# print("CHAIN===>>", chain.invoke("fatigue"))


@tool
def provide_diagnosis(symptom: str) -> str:
    """Return follow-up diagnostic questions based on the symptom to help another medical assistant make recommendations."""
    docs = vectorstore.similarity_search(symptom, k=1)  # retrieve top 1 match
    follow_up_questions = []

    for doc in docs:
        questions = doc.metadata.get("follow_up_questions", [])
        if questions:
            follow_up_questions.extend(questions)

    if follow_up_questions:
        return "I have some questions to help your diagnosis:\n- " + "\n- ".join(set(follow_up_questions))
    else:
        return f"I have no knowledge about the symptom: '{symptom}'. Please try a closely related symptom to help us discuss how you feel."

@tool
def provide_recommendation(context: str) -> str:
    """
    Generate a short, friendly, and clear medical recommendation based strictly on the patient's symptom input.
    The output must include the symptom keyword and avoid telling the patient to visit a doctor.
    The recommendation should be grounded in the dataset and limited to no more than three sentences.
    """
    prompt = (
        "You are a helpful medical assistant using only the provided dataset to make recommendations.\n"
        "Based on the symptom described below, generate a short, user-friendly recommendation.\n"
        "Do NOT suggest visiting a doctor. Do NOT invent medical facts.\n"
        f"Make sure to include the keyword: '{context.strip()}' so another assistant can explain your reasoning.\n"
        "Keep your answer to a maximum of three sentences.\n\n"
        f"Symptom: {context.strip()}\n\n"
        "Recommendation:"
    )
    response = chain.invoke(prompt)
    return response.content.strip()

@tool
def provide_explanation(context: str) -> str:
    """
    Explain in simple terms the reasoning behind a recommendation previously given,
    using only knowledge inferred from the dataset.
    The output should be user-friendly, factually grounded, and avoid speculation.
    """
    prompt = (
        "You are a healthcare assistant who explains recommendations in simple, clear terms.\n"
        "Given the context of a previous recommendation, explain the most likely reasons behind it.\n"
        "Use only knowledge from the dataset and do not speculate or invent causes.\n\n"
        f"Recommendation Context:\n{context.strip()}\n\n"
        "Explanation:"
    )
    response = chain.invoke(prompt)
    return response.content.strip()

tools = [provide_diagnosis,provide_recommendation,provide_explanation]

llm = llm.bind_tools(tools)

class MedicalAgentState(TypedDict):
  messages: Annotated[Sequence[BaseMessage],add_messages]

def medical_agent(state: MedicalAgentState) -> MedicalAgentState:
    system_prompt = SystemMessage(
        content=f"""
            You are a medical assistant responsible for managing the conversation among these worker agents: {tools}.
            - Provide diagnostic questions to examine the patient
            - Relay patient's diagnostic answers with the recommender agent to provide recommendations
            - Relay recommendations with the explainer agent to explain the reasons for the recommendation.
            - Ask the user if they need an explanations for the recommendations after recommendation is done.
            - For each response, start with the corresponding agent or tool responsible for instance (Diagnostic Agent):, (Recommendation Agent):, (Explanation Agent):.
            - Though you would discover more than one question from the diagnostic agent, ask them one at a time rather than asking them all at once.Make sure to ask all the questions and take note of the responses to provide enough information for the recommender agent and before activating it.
            - After providing a recommendation, ask the user if they need an explanation for the recommendation.
        """
    )

    print("🤖 [Agent] Invoking medical assistant with current message state...")
    response = llm.invoke([system_prompt] + state['messages'])

    # Track Tools
    print("[Agent] Tool calls detected:")
    for i, tool_call in enumerate(response.tool_calls):
        print(f" Tool #{i + 1}: {tool_call.get('name', 'UnknownTool')}")
        print(f" Args: {tool_call.get('args', {})}")
   

    return {"messages": [response]}

def should_continue(state: MedicalAgentState): 
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls: 
        return "end"
    else:
        return "continue"

graph = StateGraph(MedicalAgentState)
graph.add_node("medical_agent",medical_agent)

tool_node = ToolNode(tools=tools)

graph.add_node("tools",tool_node)

graph.set_entry_point("medical_agent")

graph.add_conditional_edges(
    "medical_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)
graph.add_edge("tools", "medical_agent")
app = graph.compile()

# Display the reAct graph architecture
display(Image(app.get_graph().draw_mermaid_png()))

# Run the app
# Conversation memory (holds the full dialogue history)
message_history: list[BaseMessage] = []

print("🤖 Welcome to your medical assistant!")
print("💬 Type your symptom or concern (e.g., 'sore throat'), or type 'exit' to quit.\n")

while True:
    user_input = input("👤 You: ").strip()

    if user_input.lower() == "exit":
        print("👋 Goodbye! Stay healthy.")
        break

    # Add the user's message to the memory
    message_history.append(HumanMessage(content=user_input))

    # Prepare state with full history
    state = {
        "messages": message_history
    }

    while True:
        # Invoke the app (agent + toolchain)
        state = app.invoke(state)

        # Retrieve and store the assistant's response
        last_message = state["messages"][-1]
        message_history.append(last_message)

        # Display the assistant response
        print("\nAssistant:")
        pprint(last_message.content)

        # End inner loop if there are no further tool calls
        if not getattr(last_message, "tool_calls", None):
            print("\nYou can ask another question or type 'exit' to quit.")
            break


