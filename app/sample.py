
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
# from langchain_openai import OpenAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import history_aware_retriever
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores.faiss import FAISS
from langchain.vectorstores import FAISS

load_dotenv()
api_key = os.getenv("OPEN_AI_KEY")
if not api_key:
    raise ValueError("OPEN_AI_KEY is not not found.")
else:
    print("OPEN_AI_KEY is loaded successfully.")

app = FastAPI()
# Define the model for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# Define the request model
class ChatRequest(BaseModel):
    question: str

# Information retrieval from the CSV file using the specified columns
print("Retrieving information from CSV file...")
def extract_csv_info(file_path):
    dataset = pd.read_csv(file_path)
    
    # Concatenate relevant columns into a single text field for each row
    dataset['combined_text'] = (
        dataset['symptom'].fillna('') + ' ' +
        dataset['conditions'].fillna('') + ' ' +
        dataset['follow_up_questions'].fillna('')
    )
    
    # Convert each row into a Document object
    documents = [Document(page_content=text) for text in dataset['combined_text'].tolist()]
    
    split_docs = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=20
    )
    splitDocs = split_docs.split_documents(documents)
    return splitDocs

print("Information retrieval completed.")
# Create a vector store from the documents
def create_vector_store(documents):
    embedding = OpenAIEmbeddings()
    vectorStore = FAISS.from_documents(documents, embedding=embedding)
    return vectorStore
docs = extract_csv_info("symptoms_data.csv")
vector_store = create_vector_store(docs)
print("Information retrieval and vector store creation completed.")
print(vector_store)