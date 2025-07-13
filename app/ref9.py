# ✅ Run the full diagnostic agent pipeline
# print("\n🚀 Running agent with input: 'I feel fatigued.'\n")
# for step in app.stream({"messages": [("user", "I feel fatigued.")]}, subgraphs=True):
#     print(step)
#     print("----")

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
