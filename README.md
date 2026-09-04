# SQL & RAG Agent using LangGraph

An intelligent question-answering agent that combines Retrieval-Augmented Generation (RAG) with natural-language SQL querying to answer questions across both unstructured documents and structured relational data.

The system is orchestrated using LangGraph, allowing the application to maintain a structured workflow and route user queries to the appropriate information source.

---

## Overview

Traditional RAG systems are well suited for answering questions from unstructured documents, while SQL agents are designed for querying structured databases.

This project combines both approaches into a single agentic workflow. The agent can determine whether a user's question requires:

* Document retrieval from a vector database
* SQL querying against a relational database
* A combination of information sources

The retrieved information is then passed to an LLM to generate a natural-language response.

---

## System Architecture

---

## Methodology

The system follows an agentic workflow consisting of several stages to evaluate incoming queries and route them to the appropriate subsystem.

### 1. User Query

The user provides a natural-language question, such as:

* *"What information is available about the company's strategy?"*
* *"What was the total revenue in 2024?"*

The system does not require the user to know SQL or the schema of the document collection.

### 2. Query Analysis

The LangGraph workflow determines what type of information is required:

* **Unstructured information** $\rightarrow$ RAG retrieval
* **Structured information** $\rightarrow$ SQL query
* **Both** $\rightarrow$ SQL + RAG

---

## RAG Pipeline

For document-based questions, the system processes unstructured data through a retrieval pipeline.

---

## SQL Agent

For structured questions, the system inspects the database schema to generate and execute an appropriate SQL query.

---

## Why LangGraph?

LangGraph is used to represent the application as a stateful graph of nodes and transitions. Instead of implementing the entire application as one large chain, individual responsibilities are represented as separate nodes.

This makes the workflow:

* **Modular** and transparent
* **Easier to debug and extend**
* **Flexible for conditional routing** across multiple tools

---

## End-to-End Workflow

---

## Project Structure

```text
SQL-and-RAG-agent-using-Langgraph/
│
├── Notebook/
│   ├── custom_agents/
│   ├── tools/
│   └── full_graph.ipynb
│
├── configs/
│
├── data/
│   └── unstructred_data/
│
├── src/
│   ├── agent_graph/
│   └── prepare_vectordb.py
│
├── agent.ipynb
├── mydb.db
├── .gitignore
└── README.md

```

---

## Technologies Used

| Technology | Purpose |
| --- | --- |
| **Python** | Core programming language |
| **LangGraph** | Agent orchestration and workflow management |
| **LangChain** | LLM and retrieval framework components |
| **LLM** | Natural-language understanding and generation |
| **SQL** | Structured relational data querying |
| **Vector Database** | Semantic document retrieval |
| **Embeddings** | Converting text into vector representations |
| **Git / GitHub** | Version control |

---

## Core Concepts Demonstrated

* **Retrieval-Augmented Generation:** Retrieves relevant external context before generating an answer, reducing reliance on the LLM's internal knowledge.
* **Natural Language to SQL:** Converts natural-language questions into SQL queries that can be executed against structured data.
* **Agentic Workflow:** Dynamically determines which information source or tool is appropriate for a given question.
* **Vector Search:** Uses semantic similarity rather than relying only on keyword matching to retrieve relevant document chunks.
* **Graph-Based Orchestration:** LangGraph represents the application as a workflow of nodes and conditional transitions.

---

## Example Use Cases

* Business intelligence assistants
* Company document analysis
* Enterprise knowledge assistants
* Financial data analysis
* Research assistants
* Internal database assistants
* Question answering over reports

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/ShefaliSingh001/SQL-and-RAG-agent-using-Langgraph.git
cd SQL-and-RAG-agent-using-Langgraph

```

### 2. Create a virtual environment

```bash
python -m venv .venv

```

Activate it:

* **macOS / Linux:** `source .venv/bin/activate`
* **Windows:** `.venv\Scripts\activate`

### 3. Install dependencies

```bash
pip install -r requirements.txt

```

### 4. Configure environment variables

Create a `.env` file containing the API credentials required by your services:

```env
GROQ_API_KEY=your_api_key

```

### 5. Prepare the vector database

Run the vector database preparation script:

```bash
python src/prepare_vectordb.py

```

### 6. Run the agent

Launch Jupyter Notebook to run the workflows:

```bash
jupyter notebook

```

Open either `agent.ipynb` or `Notebook/full_graph.ipynb`.

---

## Evaluation

A useful next step for this project is to evaluate each component independently:

| Component | Metric |
| --- | --- |
| **Query Routing** | Routing accuracy |
| **SQL Generation** | SQL execution accuracy |
| **SQL Answers** | Answer accuracy |
| **RAG Retrieval** | Precision / Recall / Hit@K |
| **RAG Generation** | Faithfulness / Correctness |
| **Overall System** | End-to-end answer accuracy |
| **Performance** | Latency / Token consumption |

---

## Future Improvements

* Add automated evaluation dataset
* Add SQL query validation & error-correction / retry logic
* Add RAG retrieval evaluation & source citations
* Add conversation memory & LangSmith tracing/observability
* Build a FastAPI backend and web interface
* Dockerise and deploy to a cloud platform
* Add automated testing & CI/CD pipeline

---

## Learning Outcomes

Through this project, I explored how modern AI systems can integrate **LLMs + RAG + SQL + Vector Search + Agentic Workflows** to build systems capable of reasoning across both structured and unstructured information sources.

---

## Author

**Shefali Singh**

Master of Information Technology

GitHub: [@ShefaliSingh001](https://www.google.com/search?q=https://github.com/ShefaliSingh001)
