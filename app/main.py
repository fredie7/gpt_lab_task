# ✅ Necessary imports
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

# ✅ Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY is missing.")
print("✅ OPENAI_API_KEY found.")

# ✅ Load and preprocess the dataset
print("\n📥 Loading and cleaning dataset...")
df = pd.read_csv("symptoms_data.csv")

# Clean column values
df['symptom'] = df['symptom'].str.strip().str.lower()
df['conditions'] = df['conditions'].apply(lambda x: [c.strip() for c in x.split(',')] if pd.notnull(x) else [])
df['follow_up_questions'] = df['follow_up_questions'].apply(lambda x: [q.strip() for q in x.split(';')] if pd.notnull(x) else [])

# print("📊 0

# ✅ Convert rows to LangChain Documents
documents = [
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

# ✅ Initialize OpenAI Embeddings
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# ✅ Create FAISS vectorstore
vectorstore = FAISS.from_documents(documents, embedding_model)
vectorstore.save_local("faiss_index")
retriever = vectorstore.as_retriever(search_type="similarity", k=3)

# ✅ Initialize Chat LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0.2, openai_api_key=OPENAI_API_KEY)
system_prompt = (
    "Use the given context to answer the question. "
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        # ("system", system_prompt),
        ("human", "Given the following context:\n\n{context}\n\nAnswer: {input}")
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, question_answer_chain)

res = chain.invoke({"input": "I feel fatigued."})
# print(res)
@tool
def retrieve_diagnosis(symptom: str):
    """Return relevant diagnostic information based on symptom."""
    results = retriever.invoke(symptom)
    responses = "\n".join([doc.page_content for doc in results])
    return f"Follow-up questions for diagnosis:\n{responses}"

@tool
def give_recommendation(context: str) -> str:
    """Use the LLM to provide a medical recommendation based on the patient's symptoms."""
    prompt = (
        "Given the patient's symptom information below, provide a 2 line short, clear medical recommendation.\n\n"
        f"Symptom Context:\n{context}\n\n"
        "Recommendation:"
    )
    response = llm.invoke(prompt)
    return response.content.strip()

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

# ✅ Workflow members
members = ["diagnostic", "recommendation", "explanation"]

class Router(TypedDict):
    next: Literal["diagnostic", "recommendation", "explanation", "FINISH"]

class State(MessagesState):
    next: str

#✅ System instruction to the router LLM
system_prompt = f"""
You are a controller managing the workflow: {members}.
Start with 'diagnostic', then 'recommendation', and finish with 'explanation'.
Once complete, choose FINISH.
"""

def supervisor_node(state: State) -> Command[Literal["diagnostic", "recommendation", "explanation", "__end__"]]:
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = llm.with_structured_output(Router).invoke(messages)
    next_step = response["next"]
    return Command(goto=END if next_step == "FINISH" else next_step, update={"next": next_step})

def diagnostic_node(state: State) -> Command[Literal["supervisor"]]:
    agent = create_react_agent(llm, tools=[retrieve_diagnosis], prompt="You are a diagnostician.")
    result = agent.invoke(state)
    return Command(update={"messages": [HumanMessage(content=result['messages'][-1].content, name="diagnostic")]}, goto="supervisor")

def recommendation_node(state: State) -> Command[Literal["supervisor"]]:
    agent = create_react_agent(llm, tools=[give_recommendation], prompt="You are a medical advisor.")
    result = agent.invoke(state)
    return Command(update={"messages": [HumanMessage(content=result['messages'][-1].content, name="recommendation")]}, goto="supervisor")

def explanation_node(state: State) -> Command[Literal["supervisor"]]:
    agent = create_react_agent(llm, tools=[explain_reasoning], prompt="You explain reasoning behind diagnoses.")
    result = agent.invoke(state)
    return Command(update={"messages": [HumanMessage(content=result['messages'][-1].content, name="explanation")]}, goto="supervisor")

# ✅ Create and compile the graph
graph = StateGraph(State)
graph.add_node("supervisor", supervisor_node)
graph.add_node("diagnostic", diagnostic_node)
graph.add_node("recommendation", recommendation_node)
graph.add_node("explanation", explanation_node)
graph.set_entry_point("supervisor")
graph.add_edge(START, "supervisor")

app = graph.compile()

# ✅ Optional: Display the graph if in Jupyter
from IPython.display import Image, display
display(Image(app.get_graph().draw_mermaid_png()))

# ✅ Run the full diagnostic agent pipeline
print("\n🚀 Running agent with input: 'I feel fatigued.'\n")
for step in app.stream({"messages": [("user", "I feel fatigued.")]}, subgraphs=True):
    print(step)
    print("----")

# ✅ Final result
result = app.invoke({"messages": [("user", "I feel fatigued.")]})
print("\n🎯 Final output:\n", result)


# if __name__ == "__main__":
#     # 🔍 Test the tool with a sample symptom
#     test_symptom = "ensure adequate rest since you are fatigued"
#     print("\n🧪 Testing retrieve_diagnosis tool with:", test_symptom)
#     output = explain_reasoning.invoke({"context": test_symptom})
#     print("\n📋 Tool Output:\n", output)



# @tool
# def explain_reasoning(context: str):
#     """Explain the diagnostic reasoning."""
#     return (
#         "This reasoning is based on symptom similarity found in medical data. "
#         "Symptoms like headache and nausea often match conditions like migraine."
#     )

# # ✅ Workflow members
# members = ["diagnostic", "recommendation", "explanation"]

# class Router(TypedDict):
#     next: Literal["diagnostic", "recommendation", "explanation", "FINISH"]

# class State(MessagesState):
#     next: str

# # ✅ System instruction to the router LLM
# system_prompt = f"""
# You are a controller managing the workflow: {members}.
# Start with 'diagnostic', then 'recommendation', and finish with 'explanation'.
# Once complete, choose FINISH.
# """

# # ✅ Supervisor node to control flow
# # def supervisor_node(state: State) -> Command:
# #     messages = [{"role": "system", "content": system_prompt}] + state["messages"]
# #     response = llm.with_structured_output(Router).invoke(messages)
# #     next_step = response["next"]
# #     return Command(goto=END if next_step == "FINISH" else next_step, update={"next": next_step})
# from langgraph.graph import END

# def supervisor_node(state: State) -> Command:
#     steps_done = state.get("steps_done", [])

#     for step in ["diagnostic", "recommendation", "explanation"]:
#         if step not in steps_done:
#             steps_done.append(step)
#             return Command(goto=step, update={"next": step, "steps_done": steps_done})

#     return Command(goto=END, update={"next": "finish", "steps_done": steps_done})



# # ✅ Diagnostic node
# # def diagnostic_node(state: State) -> Command:
# #     agent = create_react_agent(llm, tools=[retrieve_diagnosis], prompt="You are a diagnostician.")
# #     result = agent.invoke(state)
# #     return Command(update={"messages": [HumanMessage(content=result['messages'][-1].content, name="diagnostic")]}, goto="supervisor")

# def diagnostic_node(state: State) -> Command:
#     user_input = state["messages"][-1].content

#     prompt = f"""
# You are a health diagnostic assistant.
# Based on this input from the user: "{user_input}", identify the type of health problem (if any),
# such as: headache, nausea, fatigue, etc.

# Respond concisely with a sentence explaining your diagnosis.
# """

#     response = llm.invoke([HumanMessage(content=prompt)])
#     diagnosis = response.content.strip()

#     return Command(
#         update={"diagnosis": diagnosis},
#         goto="supervisor"
#     )



# # ✅ Recommendation node
# def recommendation_node(state: State) -> Command:
#     agent = create_react_agent(llm, tools=[give_recommendation], prompt="You are a medical advisor.")
#     result = agent.invoke(state)
#     return Command(update={"messages": [HumanMessage(content=result['messages'][-1].content, name="recommendation")]}, goto="supervisor")

# # ✅ Explanation node
# def explanation_node(state: State) -> Command:
#     agent = create_react_agent(llm, tools=[explain_reasoning], prompt="You explain reasoning behind diagnoses.")
#     result = agent.invoke(state)
#     return Command(update={"messages": [HumanMessage(content=result['messages'][-1].content, name="explanation")]}, goto="supervisor")

# # ✅ Create and compile the graph
# graph = StateGraph(State)
# graph.add_node("supervisor", supervisor_node)
# graph.add_node("diagnostic", diagnostic_node)
# graph.add_node("recommendation", recommendation_node)
# graph.add_node("explanation", explanation_node)
# graph.set_entry_point("supervisor")
# graph.add_edge(START, "supervisor")

# app = graph.compile()

# # ✅ Optional: Display the graph if in Jupyter
# # from IPython.display import Image, display
# # display(Image(app.get_graph().draw_mermaid_png()))

# # ✅ Run the full diagnostic agent pipeline
# print("\n🚀 Running agent with input: 'I have a headache and nausea.'\n")
# for step in app.stream({"messages": [("user", "I have a headache and nausea.")]}, subgraphs=True):
#     print(step)
#     print("----")

# # ✅ Final result
# result = app.invoke({"messages": [("user", "I have a headache and nausea.")]})
# print("\n🎯 Final output:\n", result)
