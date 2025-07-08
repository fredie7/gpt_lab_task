from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains import create_retrieval_chain

def create_agent_chain(vector_store):
    model = ChatOpenAI(model="gpt-4o", temperature=0.1)

    base_prompt = ChatPromptTemplate.from_messages([
    ("system", """
        You are a collaborative AI health assistant composed of three agents:

        1. Diagnostic Agent - Asks follow-up questions based on user symptoms starting with patient's specific details like age, gender, and current health condition.
        2. Recommendation Agent - Provides advice based on symptom patterns and likely conditions.
        3. Explanation Agent - Explains the logic behind any diagnosis or recommendation in a simple and empathetic way.

        Patient profile:
        - Age: {profile.age}
        - Gender: {profile.gender}
        - Known Conditions: {profile.known_conditions}

        Use this context:
        {context}

        Follow the steps:
        - Ask follow-up questions to clarify symptoms (Diagnostic Agent)
        - Provide suggestions based on input, user profile and conditions (Recommendation Agent)
        - Add a rationale behind your output (Explanation Agent)

        Do not group all questions in one piece so user does not feel overwhelmed. Make it take the form of a step-wise progressive conversation.

        If the user input or {input} does not relate to their health situation or case, tell the user that you are only interested in discussing their health — but do it nicely in a friendly manner.
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])


    chain = create_stuff_documents_chain(llm=model, prompt=base_prompt)

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

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
