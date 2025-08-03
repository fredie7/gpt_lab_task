#Import dependencies

# Decorator to register tools for use by agents
from langchain_core.tools import tool
# Load pre-configured vector store and retrieval chain
from data_loader import vectorstore, chain

# Define tools decorator for the agent to use
@tool
# Create a diagnostic agent that provides follow-up diagnostic questions based on the symptom to help another medical assistant make recommendations.
def provide_diagnosis(symptom: str) -> str:
    """Return follow-up diagnostic questions based on the symptom to help another medical assistant make recommendations."""

    # Retrieve top 3 similar documents related to the symptom from the vector database
    docs = vectorstore.similarity_search(symptom, k=3)  

    # Initialize an empty list to collect relevant follow-up questions.
    follow_up_questions = [] 

    # Iterate through the retrieved documents to extract follow-up questions
    for doc in docs:  
        # Get only follow-up questions from the raft of metadata, or nothing if it does not exist
        questions = doc.metadata.get("follow_up_questions", [])  
        if questions:
            # Add retrieved questions to the list
            follow_up_questions.extend(questions)  

    if follow_up_questions:
        # Return unique questions to avoid repetition
        return "I have some questions to help your diagnosis:\n- " + "\n- ".join(set(follow_up_questions)) 
    else:
        return f"I have no knowledge about the symptom: '{symptom}'. Please tell me about a closely related symptom to help us discuss how you feel."

@tool
# Create a recommendation agent that provides medical recommendations based on the user's symptom and diagnostic agent's questions.
def provide_recommendation(context: str) -> str:
    """
    Generate a short, friendly, and clear medical recommendation based strictly on the patient's symptom input.
    The output must include the symptom keyword and avoid telling the patient to visit a doctor.
    The recommendation should be grounded in the dataset and limited to no more than three sentences.
    """
    # Use the context to generate a recommendation from patient's diagnosis
    prompt = (
        "You are a helpful medical assistant using only the provided dataset to make recommendations.\n"
        "Based on the symptom described below, generate a short, user-friendly recommendation.\n"
        "Do NOT suggest visiting a doctor. Do NOT invent medical facts.\n"
        f"Make sure to include the keyword: '{context.strip()}' so another assistant can explain your reasoning.\n"
        "Keep your answer to a maximum of three sentences.\n\n"
        f"Symptom: {context.strip()}\n\n"
        "Recommendation:"
    )
    # Invoke the retrieval chain to get relevant recommendation based on the context
    response = chain.invoke(prompt)
    # Return the content of the response
    return response.content.strip()

@tool
# Create an explanation agent that explains the reasoning behind a recommendation provided by the recommender agent.
def provide_explanation(context: str) -> str:
    """
    Explain in simple terms the reasoning behind a recommendation previously given,
    using only knowledge inferred from the dataset.
    The output should be user-friendly, factually grounded, and avoid speculation.
    """
    # Use the context to generate an explanation based on the provided information by the recommendation agent
    prompt = (
        "You are a healthcare assistant who explains recommendations in simple, clear terms.\n"
        "Given the context of a previous recommendation, explain the most likely reasons behind it.\n"
        "Use only knowledge from the dataset and do not speculate or invent causes.\n\n"
        f"Recommendation Context:\n{context.strip()}\n\n"
        "Explanation:"
    )
    # Invoke the retrieval chain to include relevant explanation based on the context
    response = chain.invoke(prompt)
    # Return the content of the response
    return response.content.strip()

# List of tools for the supervisory agent to use since it is responsible for managing the conversation among the worker agents in a "ReAct" pattern.
tools = [provide_diagnosis,provide_recommendation,provide_explanation]

