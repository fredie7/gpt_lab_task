import os
import pandas as pd
import streamlit as st
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

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY is missing.")

# ---------------------------
# Load and preprocess dataset
@st.cache_data(show_spinner=True)
def load_documents():
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
    return docs

documents = load_documents()

# ---------------------------
# Setup embeddings and vectorstore with caching
@st.cache_resource(show_spinner=True)
def load_vectorstore():
    embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    FAISS_INDEX_PATH = "faiss_index"
    if os.path.exists(f"{FAISS_INDEX_PATH}/index.faiss"):
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(documents, embedding_model)
        vectorstore.save_local(FAISS_INDEX_PATH)
    return vectorstore, embedding_model

vectorstore, embedding_model = load_vectorstore()
retriever = vectorstore.as_retriever(search_type="similarity", k=3)

# ---------------------------
# LLM and QA chain setup
llm = ChatOpenAI(model="gpt-4o", temperature=0.2, openai_api_key=OPENAI_API_KEY)

prompt = ChatPromptTemplate.from_messages([
    ("human", "Given the following context:\n\n{context}\n\nAnswer: {input}")
])
qa_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, qa_chain)

# ---------------------------
# Tools

@tool
def retrieve_diagnosis(symptom: str):
    """Examine the user's health challenges and follow-up with a short question towards diagnosis"""
    results = retriever.invoke(symptom)
    responses = "\n".join([doc.page_content for doc in results])
    return f"Examine the user's health challenges and follow-up with a short question towards diagnosis:\n{responses}"

@tool
def give_recommendation(context: str) -> str:
    """Given the patient's symptom information below, provide a short but clear medical recommendation.\n\n"""
    prompt_text = (
        "Given the patient's symptom information below, provide a short but clear medical recommendation.\n\n"
        f"Symptom Context:\n{context}\n\n"
        "Recommendation:"
    )
    response = llm.invoke(prompt_text)
    return response.content.strip()

@tool
def explain_reasoning(context: str) -> str:
    """Provide explanation based on the recommendation."""
    prompt_text = (
        "Given the recommendation context below, explain the likely diagnostic reasoning in a short sentence.\n\n"
        f"Recommendation Context:\n{context}\n\n"
        "Diagnostic Reasoning:"
    )
    response = llm.invoke(prompt_text)
    return response.content.strip()

# ---------------------------
# Create agents

diagnostic_agent = create_react_agent(llm, tools=[retrieve_diagnosis], prompt="You are a diagnostician.")
recommendation_agent = create_react_agent(llm, tools=[give_recommendation], prompt="You are a medical advisor.")
explanation_agent = create_react_agent(llm, tools=[explain_reasoning], prompt="You explain reasoning behind diagnoses.")

# ---------------------------
# LangGraph workflow

members = ["diagnostic expert", "recommendation expert", "explanation expert"]

class Router(TypedDict):
    next: Literal["diagnostic", "recommendation", "explanation", "FINISH"]

class State(MessagesState):
    next: str

system_prompt = f"""
You are a controller managing the workflow from your co-workers: {members}.
Only respond with a short message from the co-worker{members} that reports back to you, one at a time and not everyone at once.
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

graph = StateGraph(State)
graph.add_node("supervisor", supervisor_node)
graph.add_node("diagnostic", diagnostic_node)
graph.add_node("recommendation", recommendation_node)
graph.add_node("explanation", explanation_node)
graph.set_entry_point("supervisor")
graph.add_edge(START, "supervisor")
result_graph = graph.compile()

# ---------------------------
# Streamlit UI

st.title("Medical Symptom Diagnosis Assistant")
st.write("Enter your symptom description below:")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_area("Your symptoms:", height=100)
submit_button = st.button("Diagnose")

if submit_button and user_input.strip():
    st.session_state.chat_history.append({"user": user_input})

    # Prepare input for LangGraph
    state_input = {"messages": [("user", user_input)]}
    with st.spinner("Analyzing..."):
        result = result_graph.invoke(state_input)

    response = result["messages"][-1].content
    st.session_state.chat_history.append({"bot": response})

# Display chat history
for entry in st.session_state.chat_history:
    if "user" in entry:
        st.markdown(f"**You:** {entry['user']}")
    if "bot" in entry:
        st.markdown(f"**Assistant:** {entry['bot']}")

