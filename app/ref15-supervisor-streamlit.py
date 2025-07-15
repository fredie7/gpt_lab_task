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
from langchain.chains import RetrievalQA
import streamlit as st


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

documents = load_documents()
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = FAISS.from_documents(documents, embedding_model)
llm = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY)
chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
# print("CHAIN===>>", chain.invoke("fatigue"))

from langchain_core.tools import tool

@tool
def retrieve_diagnosis(symptom: str) -> str:
    """Return follow-up diagnostic questions based on the symptom."""
    docs = vectorstore.similarity_search(symptom, k=1)  # retrieve top 1 match
    follow_up_questions = []

    for doc in docs:
        questions = doc.metadata.get("follow_up_questions", [])
        if questions:
            follow_up_questions.extend(questions)

    if follow_up_questions:
        return "Follow-up questions for diagnosis:\n- " + "\n- ".join(set(follow_up_questions))
    else:
        return f"I have no knowledge about the symptom: '{symptom}'. Please try a closely related symptom to help us discuss how you feel."


@tool
def give_recommendation(context: str) -> str:
    """
    Provide a short and clear medical recommendation based on the patient's symptoms.
    The response should be friendly, readable, and helpful.
    """
    prompt = (
        "You are a health assistant. Based on the symptoms described below, give a helpful, short medical recommendation "
        "in *no more than two sentences*. Make it concise and user-friendly.\n\n"
        f"Symptom: {context.strip()}\n\n"
        "Your Recommendation:"
    )

    try:
        response = llm.invoke(prompt)
        recommendation = response.content.strip()

        # Optionally format or truncate if needed
        if len(recommendation.split(".")) > 3:
            # truncate to first 2 sentences if too long
            sentences = recommendation.split(".")
            recommendation = ". ".join(sentences[:2]).strip() + "."

        return f"🩺 Recommendation:\n{recommendation}"
    except Exception as e:
        return "⚠️ Sorry, I couldn't generate a recommendation at this moment. Please try again later."


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
    "You are the Supervisor Agent responsible for managing the conversation among these worker agents: {members}."
    "diagnostic, recommendation, explanation. When a user query arrives, analyze the full conversation context—including "
    "any responses already provided by a worker—and decide which worker should handle the next task. "
    "If the user's question has already been addressed appropriately by any worker, respond with FINISH so that the same "
    "worker is not triggered again. Otherwise, if the query involves diagnostic, recommendation, explanation, "
    "choose diagnostic when the query is about symptoms; choose recommendation when the query is about treatment options; "
    "choose explanation when the query is about understanding the diagnosis. Your output must be exactly one of these words: "
    "'diagnostic', 'recommendation', 'explanation', or 'FINISH'."
"""
# """

def supervisor_node(state: State) -> Command[Literal["diagnostic", "recommendation", "explanation", "__end__"]]:
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = llm.with_structured_output(Router).invoke(messages)
    next_step = response["next"]
    print("Next step decided by supervisor:", next_step)
    return Command(goto=END if next_step == "FINISH" else next_step, update={"next": next_step})

# def diagnostic_node(state: State) -> Command[Literal["supervisor"]]:
#     result = diagnostic_agent.invoke(state)
#     return Command(update={"messages": [HumanMessage(content=result['messages'][-1].content, name="diagnostic")]}, goto="supervisor")

# def recommendation_node(state: State) -> Command[Literal["supervisor"]]:
#     result = recommendation_agent.invoke(state)
#     return Command(update={"messages": [HumanMessage(content=result['messages'][-1].content, name="recommendation")]}, goto="supervisor")

# def explanation_node(state: State) -> Command[Literal["supervisor"]]:
#     result = explanation_agent.invoke(state)
#     return Command(update={"messages": [HumanMessage(content=result['messages'][-1].content, name="explanation")]}, goto="supervisor")

def diagnostic_node(state: State) -> Command[Literal["supervisor"]]:
    result = diagnostic_agent.invoke(state)
    content = result['messages'][-1].content
    # Add agent label
    labeled_content = f"🩺 Diagnostic Agent:\n{content}"
    return Command(update={"messages": [HumanMessage(content=labeled_content, name="diagnostic")]}, goto="supervisor")

def recommendation_node(state: State) -> Command[Literal["supervisor"]]:
    result = recommendation_agent.invoke(state)
    content = result['messages'][-1].content
    labeled_content = f"💡 Recommendation Agent:\n{content}"
    return Command(update={"messages": [HumanMessage(content=labeled_content, name="recommendation")]}, goto="supervisor")

def explanation_node(state: State) -> Command[Literal["supervisor"]]:
    result = explanation_agent.invoke(state)
    content = result['messages'][-1].content
    labeled_content = f"📖 Explanation Agent:\n{content}"
    return Command(update={"messages": [HumanMessage(content=labeled_content, name="explanation")]}, goto="supervisor")


# Build and compile graph
graph = StateGraph(State)
graph.add_node("supervisor", supervisor_node)
graph.add_node("diagnostic", diagnostic_node)
graph.add_node("recommendation", recommendation_node)
graph.add_node("explanation", explanation_node)
graph.set_entry_point("supervisor")
graph.add_edge(START, "supervisor")
result_graph = graph.compile()

# print("Graph compiled successfully.")

# if __name__ == "__main__":
#     test_result = result_graph.invoke({"messages": [("user", "I feel fatigued.")]})
#     print("\nFinal output:\n", test_result['messages'][-1].content)

st.title("Symptom Diagnostic Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "user", "content": ""}]

def submit_query():
    user_input = st.session_state.user_input.strip()
    if user_input:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Prepare messages for graph input format
        messages_for_graph = [(msg["role"], msg["content"]) for msg in st.session_state.messages if msg["content"]]

        # Invoke the LangGraph workflow
        result = result_graph.invoke({"messages": messages_for_graph})

        # Extract last message content from the workflow output
        last_message = result['messages'][-1].content

        # Append assistant reply
        st.session_state.messages.append({"role": "assistant", "content": last_message})

        # Clear input box
        st.session_state.user_input = ""

# Input box and submit button
st.text_input("Describe your symptom or ask a question:", key="user_input", on_change=submit_query)

# Display conversation history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Bot:** {msg['content']}")
