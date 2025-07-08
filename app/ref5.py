from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.docstore.document import Document

import asyncio

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not found.")

# FastAPI app setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class ChatRequest(BaseModel):
    question: str

# Extract and preprocess documents
def extract_csv_info(file_path):
    dataset = pd.read_csv(file_path)
    dataset['combined_text'] = (
        dataset['symptom'].fillna('') + ' ' +
        dataset['conditions'].fillna('') + ' ' +
        dataset['follow_up_questions'].fillna('')
    )
    documents = [Document(page_content=text) for text in dataset['combined_text'].tolist()]
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
    return splitter.split_documents(documents)

# Create vector store
def create_vector_store(documents):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(documents, embedding=embeddings)

# Define each agent prompt and model
def create_agent_prompt(system_message: str):
    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

# Diagnostic agent prompt & model
diagnostic_prompt_text = """
You are the Diagnostic Agent. Your task is to ask follow-up questions based on user symptoms.
Begin by warmly gathering the patient's age, gender, and current health status in a friendly manner.
Ask questions progressively, do not overwhelm with many questions at once.
"""
diagnostic_prompt = create_agent_prompt(diagnostic_prompt_text)
diagnostic_model = ChatOpenAI(model="gpt-4o", temperature=0.1)

# Recommendation agent prompt & model
recommendation_prompt_text = """
You are the Recommendation Agent. Based on the symptoms, user profile, and conditions, provide advice.
Be clear and concise in your recommendations.
"""
recommendation_prompt = create_agent_prompt(recommendation_prompt_text)
recommendation_model = ChatOpenAI(model="gpt-4o", temperature=0.1)

# Explanation agent prompt & model
explanation_prompt_text = """
You are the Explanation Agent. Explain the reasoning behind the diagnosis or recommendations simply and empathetically.
"""
explanation_prompt = create_agent_prompt(explanation_prompt_text)
explanation_model = ChatOpenAI(model="gpt-4o", temperature=0.1)

# Wrap the agents into async functions
async def run_agent(agent_model, agent_prompt, user_input, chat_history):
    formatted_prompt = agent_prompt.format_prompt(chat_history=chat_history, input=user_input)
    response = await agent_model.ainvoke(formatted_prompt.to_messages())
    return response.content


# Multi-agent controller to handle state and flow
class MultiAgentController:
    def __init__(self, vector_store):
        self.chat_history = []  # List of HumanMessage/AIMessage for context
        self.diagnostic_done = False
        self.vector_store = vector_store

    async def handle_user_input(self, user_input: str) -> str:
        # Add user input to chat history
        self.chat_history.append(HumanMessage(content=user_input))

        if not self.diagnostic_done:
            diag_response = await run_agent(diagnostic_model, diagnostic_prompt, user_input, self.chat_history)
            self.chat_history.append(AIMessage(content=diag_response))

            # Simple heuristic: if user says they are done with questions
            if any(phrase in user_input.lower() for phrase in ["no more questions", "done", "that's all"]):
                self.diagnostic_done = True
                return "Thank you for the information. I will now provide recommendations based on your inputs. Do you mind?"

            return diag_response

        # After diagnostics done: Recommendation + Explanation
        # Summarize diagnostic info from chat_history for retrieval
        diagnostic_texts = " ".join(msg.content for msg in self.chat_history if isinstance(msg, AIMessage))
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        related_docs = retriever.get_relevant_documents(diagnostic_texts)
        docs_text = "\n".join(doc.page_content for doc in related_docs)

        recommendation_input = f"Symptoms and diagnostic info:\n{diagnostic_texts}\n\nRelevant medical info:\n{docs_text}"
        recommendation_response = await run_agent(recommendation_model, recommendation_prompt, recommendation_input, self.chat_history)
        self.chat_history.append(AIMessage(content=recommendation_response))

        explanation_response = await run_agent(explanation_model, explanation_prompt, recommendation_response, self.chat_history)
        self.chat_history.append(AIMessage(content=explanation_response))

        answer = f"{recommendation_response}\n\nExplanation:\n{explanation_response}"
        return answer

# Initialize
csv_file_path = "symptoms_data.csv"  # Make sure this file exists with the proper columns
documents = extract_csv_info(csv_file_path)
vector_store = create_vector_store(documents)
controller = MultiAgentController(vector_store)

# API endpoint
@app.post("/ask")
async def chat(request: ChatRequest):
    try:
        answer = await controller.handle_user_input(request.question)
        return {"response": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run app with: uvicorn your_script_name:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
