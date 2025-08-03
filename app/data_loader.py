# Import dependencies

# Pandas for data manipulation
import pandas as pd
# To load environment variables from the .env file
from dotenv import load_dotenv
# To represent text chunks with metadata
from langchain_core.documents import Document
# To split large text into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
# To load OpenAI tools for embeddings and LLM
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# To facilitate similarity search
from langchain_community.vectorstores import FAISS
# To faciitate RAG pipeline for response retrieval
from langchain.chains import RetrievalQA
# To access environment variables
import os


# Load environment variables from .env file
load_dotenv()

# Fetch the OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing.")
print("OPENAI_API_KEY found.")

# Load and preprocess dataset
def load_documents():

    # Use caching so the function only loads once
    if hasattr(load_documents, "cached"):
        return load_documents.cached

    print("Loading and cleaning dataset...")

    # Read the dataset
    dataset = pd.read_csv("symptoms_data.csv")

    # Enlist the column names
    dataset_columns = ['symptom', 'conditions', 'follow_up_questions']

    # Check for missing values in the dataset's columns
    for col in dataset_columns:
        if dataset[col].isnull().any():
            print(f"The column '{col}' has null values.")
        else:
            print(f"The column '{col}' does not have a null value.")

    # Clean and normalize the data

    # Remove white spaces and revert to lowercases
    dataset['symptom'] = dataset['symptom'].str.strip().str.lower()
    dataset['conditions'] = dataset['conditions'].apply(lambda x: [c.strip().lower() for c in x.split(',')])
    dataset['follow_up_questions'] = dataset['follow_up_questions'].apply(lambda x: [q.strip().lower() for q in x.split(';')])

    # Convert each row into a LangChain Document with metadata
    docs = [
        Document(
            page_content=f"symptom: {row['symptom']}\nconditions: {', '.join(row['conditions'])}\nfollow_up: {'; '.join(row['follow_up_questions'])}",
            metadata={
                "symptom": row['symptom'],
                "conditions": row['conditions'],
                "follow_up_questions": row['follow_up_questions']
            }
        )
        for _, row in dataset.iterrows()
    ]
    load_documents.cached = docs
    return docs

# Perform RAG(Retrieval-Augmented Generation)

# Load documents
documents = load_documents()


# Create vector embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)

# Split documents into overlapping chunks to ensure context isn't lost
text_splitter = RecursiveCharacterTextSplitter(
    # Max chunk size in characters
    chunk_size=700,
    # Amount of overlap between chunks to preserves context
    chunk_overlap=500
)

# Split documents into smaller chunks for better retrieval to help the model retrieve relevant context
documents_split = text_splitter.split_documents( documents)

# Create a vector store using FAISS to store the  documents within specified index for swift similarity search
vectorstore = FAISS.from_documents(documents_split, embeddings)

# Initialize the LLM and set the temperature in a way to make the model more deterministic and prevent hallucination
llm = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY,temperature=0.0)

# Create a retrieval chain to handle the retrieval of relevant documents and generate responses based on user input
chain = RetrievalQA.from_chain_type(llm=llm, chain_type="map_reduce", retriever=vectorstore.as_retriever())
# print("CHAIN===>>", chain.invoke("fatigue"))
