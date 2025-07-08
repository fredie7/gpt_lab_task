from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import pandas as pd
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langgraph.graph import StateGraph
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not found.")

print("\n📥 Loading dataset...")
df = pd.read_csv("symptoms_data.csv")

# Convert rows into LangChain Documents
docs = []
for _, row in df.iterrows():
    doc = Document(
        page_content=f"Symptom: {row['symptom']}. Conditions: {row['conditions']}. Questions: {row['follow_up_questions']}",
        metadata={
            "symptom": row["symptom"],
            "conditions": row["conditions"],
            "follow_up": row["follow_up_questions"]
        }
    )
    docs.append(doc)
print("✅ Loaded", len(docs), "documents.")

def diagnostic_agent(state):
    user_input = state["input"]
    for _, row in df.iterrows():
        if row["symptom"].lower() in user_input.lower():
            follow_up = row["follow_up_questions"].split(";")[0]
            return {"follow_up": follow_up, "stage": "recommend"}
    return {"follow_up": "Can you describe your symptom more?", "stage": "diagnose"}

# Node: Recommend
def recommendation_agent(state):
    return {
        "recommendation": "Try resting in a dark, quiet room and consider over-the-counter pain relief.",
        "stage": "explain"
    }

# Node: Explain
def explanation_agent(state):
    return {
        "explanation": "This is based on your symptom matching common migraine patterns.",
        "stage": "done"
    }

# ===========================
# 🧩 Step 5: LangGraph Setup
# ===========================

from typing import TypedDict, Optional, Literal
from langgraph.graph import StateGraph, END
from IPython.display import Image, display

# Define the structure of the conversation state
class DiagnosticState(TypedDict):
    input: str
    stage: Literal["diagnose", "recommend", "explain", "done"]
    follow_up: Optional[str]
    recommendation: Optional[str]
    explanation: Optional[str]

# Initialize the graph
builder = StateGraph(DiagnosticState)

# Add your agents (functions)
builder.add_node("Diagnose", diagnostic_agent)
builder.add_node("Recommend", recommendation_agent)
builder.add_node("Explain", explanation_agent)

# Start at Diagnose
builder.set_entry_point("Diagnose")

# Decide which node to go to next
def next_node(state: DiagnosticState) -> str:
    return state["stage"]

# Add transitions
builder.add_conditional_edges("Diagnose", next_node, {
    "diagnose": "Diagnose",      # More follow-up needed
    "recommend": "Recommend"     # Ready to move to recommendations
})

builder.add_conditional_edges("Recommend", next_node, {
    "explain": "Explain"
})

# ✅ Instead of using `None`, use the `END` marker here
builder.add_conditional_edges("Explain", next_node, {
    "done": END
})

# Compile the graph
app_graph = builder.compile()

# Optional: Display graph structure visually
# display(Image(app.get_graph().draw_mermaid_png()))

app = FastAPI()

# Enable CORS (for frontend use, optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class UserMessage(BaseModel):
    question: str

# Response model
class BotResponse(BaseModel):
    answer: str

# Route
@app.post("/ask", response_model=BotResponse)
async def chat(user_input: UserMessage):
    try:
        initial_state = {"input": user_input.question, "stage": "diagnose"}
        final_state = app_graph.invoke(initial_state)

        # Prioritize most complete output as the answer
        if "explanation" in final_state:
            return {"answer": final_state["explanation"]}
        elif "recommendation" in final_state:
            return {"answer": final_state["recommendation"]}
        elif "follow_up" in final_state:
            return {"answer": final_state["follow_up"]}
        else:
            return {"answer": "Sorry, I couldn't understand your symptom. Please try again."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)