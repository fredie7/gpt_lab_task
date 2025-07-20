## Agentic AI for Medical Diagnosis, Recommendation & Explanation

This project delivers an Agentic AI Healthcare Assistant that integrates Retrieval-Augmented Generation (RAG) with Reasoning and Action (ReAct) architecture.
The workflow involves a supervisory agent which engages in a continuous feedback loop with the:
<li>Diagnostic agent</li>
<li>Recommender agent</li>
<li>Explainer agent</li>
This collaborative framework enables the assistant to not only provide diagnostic insights but also offer personalized recommendations accompanied by supportive explanations.


### Tools Used:
Python, Langchain, Langgraph, Next Js

### The Retrieval-Augmented Generation(RAG) Pipeline

The project's RAG pipeline begins with the preprocessing of medical data, which includes symptoms, associated conditions, and follow-up questions. To ensure the integrity of the data and prevent leakage during interaction with a large language model (LLM), the dataset is first examined for inconsistencies, procesed, then divided into manageable chunks. This is accomplished using the RecursiveCharacterTextSplitter, configured with a chunk size of 1000 and a chunk overlap of 200.
Next, a FAISS (Facebook AI Similarity Search) vector database is introduced to store the processed document chunks. To enable efficient similarity search, the text chunks are first converted into numerical embeddings using an OpenAI embedding utility referred to as “text-embedding-3-smal“. These embeddings are then stored in the FAISS vector database in index form for fast retrieval.
Afterwards, a retrieval mechanism is established to fetch the most relevant document chunks in response to prospective user queries. This is achieved through the RetrievalQA component, which integrates the retriever with a language model. The retriever identifies and pulls the most relevant context from the FAISS index, while the language model. OpenAI’s GPT-4o generates a coherent response. A temperature setting of 0.0 is used to minimize hallucinations and ensure consistent output.


### System Design

The system design follows the classic ReAct architecture, which integrates reasoning and action in a crisp decision-making process. At the core of this framework is a master agent that utilizes a Large Language Model to reason through problems. This agent is also equipped with predefined tools, also known as functions, which it uses to perform specific actions. The process begins with the master agent analyzing the problem and selecting the most appropriate tool from a suite of available options. Once the selected tool performs its task, the result is returned to the agent. The agent then combines this result with its understanding of the task to generate a final output for the user.

<!--![image_alt](https://github.com/fredie7/gpt_lab_task/blob/main/Screenshot%20(3736).png?raw=true)-->

<div align="center">
  <img src="https://github.com/fredie7/gpt_lab_task/blob/main/System%20design%20(3736).png?raw=true" />
  <br>
   <sub><b>Fig 1.</b> Work FLow</sub>
</div>
  


Similarly, this health-care assistant employs a master supervisory agent that initiates the application flow. This agent operates with a low-temperature LLM to minimize hallucinations and ensure stable outputs. It is bound to three specialized tools that incorporate the retrieval augmented generation pipeline in their working functions, and these tools are: a diagnostic tool, a recommendation tool, and an explanatory tool. The diagnostic tool asks probing questions about the user's health, based on predefined data. The recommendation tool offers appropriate suggestions, while the explanatory tool provides justifications for those suggestions.
As illustrated in the system diagram, the agent serves as the starting point of the application. It engages in a feedback loop with the tools node, which encapsulates all three tools. Each tool contains a key called “tool_call”, which signals whether it has information to return to the master agent. This feedback loop, represented by the “continue” edge, remains active as long as there are pending “tool_calls”. Once all tool calls are completed, the process transitions back to the master agent, through the upper “end” edge to the lower “end” node, where the final response is delivered to the user.

### Agent Collboration

<table align="center">
  <tr>
    <td align="center">
      <img src="https://github.com/fredie7/gpt_lab_task/blob/main/Tool%20Calls%20(3770).png?raw=true" height="300"><br>
      <sub><b>Fig 2.</b> Tool Calls</sub>
    </td>
    <td align="center">
      <img src="https://github.com/fredie7/gpt_lab_task/blob/main/Tools%20interaction%20(3774).png?raw=true" height="300"><br>
      <sub><b>Fig 3.</b> Tools Interaction</sub>
    </td>
  </tr>
</table>





