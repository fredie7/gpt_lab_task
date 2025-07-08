import pandas as pd
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings

def build_vector_store(csv_path: str):
    df = pd.read_csv(csv_path)
    df.fillna('', inplace=True)

    df['combined_text'] = df['symptom'] + '. ' + df['conditions'] + '. ' + df['follow_up_questions']
    documents = [Document(page_content=row) for row in df['combined_text'].tolist()]

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    split_docs = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(split_docs, embedding=embeddings)
    return vector_store
