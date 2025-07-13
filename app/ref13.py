import os
import random
import pandas as pd
from dotenv import load_dotenv
from typing import Literal
from typing_extensions import TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langgraph.graph import MessageGraph, END
from langgraph.types import Command
from langgraph.graph.message import MessagesState

from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing.")
print("✅ OPENAI_API_KEY loaded")

# FastAPI app + CORS config
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SymptomInput(BaseModel):
    message: str

def load_documents():
    if hasattr(load_documents, "cached"):
        return load_documents.cached

    print("📥 Loading and preprocessing symptoms dataset...")
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

documents = load_documents()
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = FAISS.from_documents(documents, embedding_model)
llm = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY)
chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

# Supervisor config
members = ["diagnoser", "recommender", "explainer"]
system_prompt = (
    "You are the Supervisor Agent responsible for managing the conversation among these worker agents: "
    "diagnoser, recommender, and explainer. Analyze the full conversation context. "
    "Decide which worker should handle the next task. Choose 'diagnoser', 'recommender', 'explainer', or 'FINISH'."
)

class Router(TypedDict):
    next: Literal["diagnoser", "recommender", "explainer", "FINISH"]

class State(MessagesState):
    next: str

def supervisor_node(state: State) -> Command[str]:
    messages = [
        {"role": "system", "content": system_prompt}
    ] + [m.to_dict() for m in state["messages"] if isinstance(m, BaseMessage)]

    response = llm.with_structured_output(Router).invoke(messages)
    goto = response["next"]
    return Command(goto=END if goto == "FINISH" else goto, update={"next": goto})

def diagnoser_agent(state: State) -> State:
    user_input = state["messages"][-1].content
    context_docs = chain.get_relevant_documents(user_input)

    questions = []
    for doc in context_docs:
        qlist = doc.metadata.get("follow_up_questions", [])
        questions.extend(qlist)

    followups = random.sample(questions, min(2, len(questions))) if questions else [
        "Could you tell me more about your symptoms?"
    ]
    msg = AIMessage(content="Thank you. To clarify, could you answer the following:\n- " + "\n- ".join(followups))
    return {"messages": state["messages"] + [msg], "next": ""}

def recommender_agent(state: State) -> State:
    user_input = state["messages"][-1].content
    context_docs = chain.get_relevant_documents(user_input)

    suggestions = []
    for doc in context_docs:
        for condition in doc.metadata.get("conditions", []):
            suggestions.append(
                f"For {condition.strip()}, consider adequate rest, hydration, and consulting a specialist if symptoms persist."
            )

    response = "\n\n".join(set(suggestions)) if suggestions else "Please consult a healthcare provider for personalized recommendations."
    msg = AIMessage(content=response)
    return {"messages": state["messages"] + [msg], "next": ""}

def explainer_agent(state: State) -> State:
    explanation = (
        "Based on similar symptom patterns in our medical database, these conditions were considered likely matches. "
        "The recommendations were made to manage symptoms and promote recovery while preventing complications."
    )
    msg = AIMessage(content=explanation)
    return {"messages": state["messages"] + [msg], "next": ""}

# Graph setup
graph = MessageGraph()
graph.add_node("diagnoser", diagnoser_agent)
graph.add_node("recommender", recommender_agent)
graph.add_node("explainer", explainer_agent)
graph.add_node("supervisor", supervisor_node)
graph.set_entry_point("supervisor")
graph.add_conditional_edges("supervisor", lambda state: state["next"], {
    "diagnoser": "diagnoser",
    "recommender": "recommender",
    "explainer": "explainer",
    END: END
})

for member in members:
    graph.add_edge(member, "supervisor")

runnable = graph.compile()

# Manual test
if __name__ == "__main__":
    print("\n🩺 Running multi-agent test...\n")
    initial_state = {
        "messages": [HumanMessage(content="I have a persistent fever")],
        "next": ""
    }
    for output in runnable.stream(initial_state):
        print("----")
        last_msg = output["messages"][-1]
        print(f"{last_msg.__class__.__name__}: {last_msg.content}")
