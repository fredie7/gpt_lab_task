## Agentic AI for Medical Diagnosis, Recommendation & Explanation

This project delivers an Agentic AI Healthcare Assistant that integrates Retrieval-Augmented Generation (RAG) with Reasoning and Action (ReAct) architecture. This collaborative framework enables the assistant to not only provide diagnostic insights but also offer personalized recommendations accompanied by supportive explanations.

### The RAG Pipeline

The Retrieval-Augmented Generation (RAG) pipeline begins with the preprocessing of medical data, which includes symptoms, associated conditions, and follow-up questions. To ensure the integrity of the data and prevent leakage during interaction with a large language model (LLM), the dataset is first divided into manageable chunks. This is accomplished using the RecursiveCharacterTextSplitter, configured with a chunk size of 1000 and a chunk overlap of 200.
Next, a FAISS (Facebook AI Similarity Search) vector database is introduced to store the processed document chunks. To enable efficient similarity search, the text chunks are first converted into numerical embeddings using an OpenAI embedding utility referred to as “text-embedding-3-smal“. These embeddings are then stored in the FAISS vector database in index form for fast retrieval.
Afterwards, a retrieval mechanism is established to fetch the most relevant document chunks in response to user queries. This is achieved through the RetrievalQA component, which integrates the retriever with a language model. The retriever identifies and pulls the most relevant context from the FAISS index, while the language model. OpenAI’s GPT-4o generates a coherent response. A temperature setting of 0.0 is used to minimize hallucinations and ensure consistent output.


### System Design
![image_alt](https://github.com/fredie7/gpt_lab_task/blob/main/Screenshot%20(3736).png?raw=true)
