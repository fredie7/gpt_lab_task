## Agentic AI for Medical Diagnosis, Recommendation & Explanation

This project delivers an Agentic AI Healthcare Assistant that integrates Retrieval-Augmented Generation (RAG) with Reasoning and Action (ReAct) architecture.
The workflow involves a supervisory agent which engages in a continuous feedback loop with the:
<li>Diagnostic agent</li>
<li>Recommender agent</li>
<li>Explaination agent</li>
This collaborative framework enables the assistant to not only provide diagnostic insights but also offer personalized recommendations accompanied with supportive explanations.


### Tools Used:
Python, Langchain, Langgraph, Next Js

### The Retrieval-Augmented Generation(RAG) Pipeline

The project's RAG pipeline begins with the preprocessing of medical data, which includes symptoms, associated conditions, and follow-up questions. To ensure the integrity of the data and prevent leakage during interaction with a large language model (LLM), the dataset is first examined for inconsistencies, procesed, then divided into manageable chunks. This is accomplished using the RecursiveCharacterTextSplitter, configured with a chunk size of 700 and a chunk overlap of 500.
Next, a FAISS (Facebook AI Similarity Search) vector database is introduced to store the processed document chunks. To enable efficient similarity search, the text chunks are first converted into numerical embeddings using an OpenAI embedding utility referred to as “text-embedding-3-smal“. These embeddings are then stored in the FAISS vector database in index form for fast retrieval.
Afterwards, a retrieval mechanism is established to fetch the most relevant document chunks in response to prospective user queries. This is achieved through the RetrievalQA component, which integrates the retriever with a language model. The retriever identifies and pulls the most relevant context from the FAISS index, while the language model. OpenAI’s GPT-4o generates a coherent response. A temperature setting of 0.0 is used to minimize hallucinations and ensure consistent output.


### System Design

The system design follows the classic ReAct architecture, which integrates reasoning and action in a crisp decision-making process. At the core of this framework is a master agent that utilizes a Large Language Model to reason through problems. This agent is also equipped with predefined tools, also known as functions, which it uses to perform specific actions. The process begins with the master agent analyzing the problem and selecting the most appropriate tool from a suite of available options. Once the selected tool performs its task, the result is returned to the agent. The agent then combines this result with its understanding of the task to generate a final output for the user.

<!--![image_alt](https://github.com/fredie7/gpt_lab_task/blob/main/Screenshot%20(3736).png?raw=true)-->

<div align="center">
  <img src="https://github.com/fredie7/gpt_lab_task/blob/main/Syatem%20Design%20(3778).png?raw=true" />
  <br>
   <sub><b>Fig 1.</b> Workflow</sub>
</div>
  


Similarly, this health-care assistant employs a master supervisory agent that initiates the application flow. This agent operates with a low-temperature LLM to minimize hallucinations and ensure stable outputs. It is bound to three specialized tools that incorporate the retrieval augmented generation pipeline in their working functions, and these tools are: a diagnostic tool, a recommendation tool, and an explanatory tool. The diagnostic tool asks probing questions about the user's health, based on predefined data. The recommendation tool offers appropriate suggestions, while the explanatory tool provides justifications for those suggestions.
As illustrated in the workflow from (Fig 1), the medical agent serves as the starting point of the application. It engages in a feedback loop with the tools node, which encapsulates all three tools. Each tool contains a key called “tool_call”, which signals whether it has information to return to the master agent. This feedback loop, represented by the “continue” edge, remains active as long as there are pending “tool_calls”. Once all tool calls are completed, the process transitions back to the master agent, through the upper “end” edge to the lower “end” node, where the final response is delivered to the user.

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

The agentic AI workflow follows a continuous loop of reasoning and action between the medical agent and its three sub-agents. As shown in Fig. 2 and Fig. 3, the diagnostic agent is labeled as provide_diagnosis, the recommender agent as provide_recommendation, and the explanation agent as provide_explanation.

In Fig. 2, the first arrow indicates the initial step, where the medical agent requests feedback from all three sub-agents after receiving a user message. The second arrow points to the activation of the provide_diagnosis tool, followed by the third and fourth arrows, which point to the provide_recommendation and provide_explanation tools, respectively.

These tools are triggered in response to the flow of conversation—depending on whether the user seeks a diagnosis for a health concern, a medical recommendation or advice, or an explanation for a previously provided suggestion.

The first arrow in Fig. 3 shows how the user’s message is initially collected and passed as an argument to the diagnostic agent. The steps that follow the second arrow involve continuous interactions between the diagnostic agent and the user, during which the agent gathers all necessary information. Once this process is complete, the second arrow indicates the point at which the recommender agent is activated. The third arrow then shows how the recommender agent relays its report to the explanation agent, along with contextual user data to support the explanation process.

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/fredie7/gpt_lab_task/blob/main/UI%20(3776).png?raw=true" height="200"><br>
      <sub><b>Fig 4.</b> UI Landing Page</sub>
    </td>
    <td align="center">
      <img src="https://github.com/fredie7/gpt_lab_task/blob/main/conversation_1%20(3763).png?raw=true" height="200"><br>
      <sub><b>Fig 5.</b>Diagnostic Conversation</sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/fredie7/gpt_lab_task/blob/main/conversation_2%20(3764).png?raw=true" height="200"><br>
      <sub><b>Fig 6.</b>Diagnostic Conversation</sub>
    </td>
    <td align="center">
      <img src="https://github.com/fredie7/gpt_lab_task/blob/main/conversation_3%20(3765).png?raw=true" height="200"><br>
      <sub><b>Fig 7.</b>Diagnostic Conversation</sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/fredie7/gpt_lab_task/blob/main/conversation_4%20(3766).png?raw=true" height="200"><br>
      <sub><b>Fig 8.</b>Diagnostic & Recommendation Conversation 4</sub>
    </td>
    <td align="center">
      <img src="https://github.com/fredie7/gpt_lab_task/blob/main/conversation_5%20(3767).png?raw=true" height="200"><br>
      <sub><b>Fig 9.</b>Recommendation & Explanation Conversation 5</sub>
    </td>
  </tr>
</table>

The Fig 4 above shows the user interface of the application, while Fig 5 to Fig 9 show the interaction between the sample user, “Rico” and all 3 agents (Diagnostic Agent, Recommender Agent & Explainer Agent)

To run the system locally on your computer:
1.	Spin up your terminal from Visual Studio Code
2.	Navigate to the Command Prompt(preferred)
3.	Clone the repository using: git clone https://github.com/fredie7/gpt_lab_task.git
4.	Go into the directory: cd gpt_lab_task
5.	Create a virtual environment: python -m venv env
6.	Go into the virtual environment: env\Scripts\activate.bat
7.	Go into the app directory cd app
8.	Download the requirements.txt file: pip install -r requirements.txt
9.	Start the application: uvicorn main:app –reload
10.	Wait for a few seconds for the notification “Application startup complete.”

Meanwhile, here's a link to the frontend repository: (https://github.com/fredie7/gpt_lab_task_frontend)



