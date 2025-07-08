import streamlit as st
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

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not found.")

@st.cache_data
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

@st.cache_resource
def create_vector_store(_documents):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(_documents, embedding=embeddings)

@st.cache_resource
def create_agent_chain(_vector_store):
    model = ChatOpenAI(model="gpt-4o", temperature=0.1)

    base_prompt = ChatPromptTemplate.from_messages([
        ("system", """
            You are a collaborative AI health assistant composed of three agents:

            1. Diagnostic Agent - Asks follow-up questions based on user symptoms starting with patient's specific details like age, gender, and current health condition.
            2. Recommendation Agent - Provides advice based on symptom patterns and likely conditions.
            3. Explanation Agent - Explains the logic behind any diagnosis or recommendation in a simple and empathetic way.

            Use this context:
            {context}

            Follow the steps:
            - Ask follow-up questions to clarify symptoms (Diagnostic Agent)
            - Provide suggestions based on input, user profile and conditions (Recommendation Agent)
            - Add a rationale behind your output (Explanation Agent)

            "Do not group all questions in one piece so user does not feel overwhelmed. Make it take the form of a step-wise progressive conversation"
            "If the user input or message or {input} does not relate to their health situation or case, tell the user that ‘I am only interested in discussing their health' but do it nicely in a friendly manner "
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    chain = create_stuff_documents_chain(llm=model, prompt=base_prompt)
    retriever = _vector_store.as_retriever(search_kwargs={"k": 3})

    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm=model,
        retriever=retriever,
        prompt=retriever_prompt
    )

    return create_retrieval_chain(history_aware_retriever, chain)


# Load and prepare data once (cached)
csv_file_path = "symptoms_data.csv"
documents = extract_csv_info(csv_file_path)
vector_store = create_vector_store(documents)
retrieval_chain = create_agent_chain(vector_store)

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("RAG-Driven Diagnostic AI Assistant")

# User input
user_question = st.text_input("Describe your symptoms:")

if st.button("Ask") and user_question:
    try:
        result = retrieval_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input": user_question
        })
        answer = result["answer"]
        # Append to chat history
        st.session_state.chat_history.append(HumanMessage(content=user_question))
        st.session_state.chat_history.append(AIMessage(content=answer))

        # Display conversation
        st.markdown(f"**You:** {user_question}")
        st.markdown(f"**AI:** {answer}")

    except Exception as e:
        st.error(f"Error: {e}")

# Optionally display chat history
if st.checkbox("Show chat history"):
    for msg in st.session_state.chat_history:
        role = "You" if isinstance(msg, HumanMessage) else "AI"
        st.markdown(f"**{role}:** {msg.content}")
