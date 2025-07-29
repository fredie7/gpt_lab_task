import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os


# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load and preprocess dataset once
def load_documents():
    if hasattr(load_documents, "cached"):
        return load_documents.cached

    print("Loading and cleaning dataset...")
    df = pd.read_csv("symptoms_data.csv")
    dataset_columns = ['symptom', 'conditions', 'follow_up_questions']

    for col in dataset_columns:
        if df[col].isnull().any():
            print(f"The column '{col}' has null values.")
        else:
            print(f"The column '{col}' does not have a null value.")

    df['symptom'] = df['symptom'].str.strip().str.lower()
    df['conditions'] = df['conditions'].apply(lambda x: [c.strip().lower() for c in x.split(',')])
    df['follow_up_questions'] = df['follow_up_questions'].apply(lambda x: [q.strip().lower() for q in x.split(';')])

    docs = [
        Document(
            page_content=f"symptom: {row['symptom']}\nconditions: {', '.join(row['conditions'])}\nfollow_up: {'; '.join(row['follow_up_questions'])}",
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

# Perform RAG(Retrieval-Augmented Generation)

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)

# Split documents into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=500
)

# Split documents into smaller chunks for better retrieval to help the model retrieve relevant context
documents_split = text_splitter.split_documents( documents)

# Create a vector store using FAISS to store the  documents within specified index
vectorstore = FAISS.from_documents(documents_split, embeddings)

# Initialize the LLM and set the temperature in a way to prevent hallucination
llm = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY,temperature=0.0)

# Create a retrieval chain to handle the retrieval of relevant documents and generate responses based on user input
chain = RetrievalQA.from_chain_type(llm=llm, chain_type="map_reduce", retriever=vectorstore.as_retriever())
# print("CHAIN===>>", chain.invoke("fatigue"))
