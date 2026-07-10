# 🎓 EduAI: Agentic RAG College Admission Assistant

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-black?logo=flask)
![LangChain](https://img.shields.io/badge/LangChain-Orchestration-green)
![IBM Watsonx](https://img.shields.io/badge/IBM%20Watsonx-Granite%20LLM-0f62fe)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Database-orange)

## 📌 Project Overview

**EduAI** is an intelligent, Agentic Retrieval-Augmented Generation (RAG) system designed to autonomously streamline the college admission process. Built for prospective engineering students, this digital assistant retrieves highly specific institutional data—such as B.Tech CSE eligibility criteria, tuition fees, course matrices, and hostel details—and synthesizes factual, real-time responses using IBM's enterprise-grade Granite models. 

This project was developed in alignment with the **AICTE & IBM University Engagement** challenge (Problem Statement No.4).

---

## 🚀 Key Features

* **Zero-Hallucination Answers:** Utilizes a strict RAG pipeline to ensure the AI only speaks based on verified institutional data.
* **Semantic Search:** Employs a local FAISS vector database and HuggingFace embeddings (`all-MiniLM-L6-v2`) to perform high-speed similarity searches across unstructured text.
* **Enterprise Intelligence:** Powered by **IBM watsonx.ai** (`ibm-granite-3-1-8b-base`) for advanced natural language reasoning and contextual understanding.
* **Conversational Memory:** Retains chat history within the session to allow for natural, context-aware follow-up questions.
* **Responsive Web UI:** Features a sleek, dark-mode user interface built with Python Flask, Bootstrap, and asynchronous JavaScript.

---

## 🧠 System Architecture

1. **Data Ingestion:** Institutional knowledge (`university_data.txt`) is processed and chunked using `RecursiveCharacterTextSplitter`.
2. **Vectorization:** Text is converted into high-dimensional vectors and stored locally in a FAISS index.
3. **Retrieval:** User queries are embedded and compared against the vector database to retrieve the top contextual matches.
4. **Generation:** The retrieved facts and user prompt are securely passed to the IBM Granite LLM to formulate a precise response.

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/College-Admission-Agent.git
cd College-Admission-Agent
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Rename the `.env.example` file to `.env` and insert your IBM Cloud credentials:
```env
IBM_CLOUD_API_KEY=your_ibm_cloud_api_key_here
PROJECT_ID=your_watsonx_project_id_here
IBM_WATSONX_URL=https://us-south.ml.cloud.ibm.com
```

### 5. Run the Application
```bash
python app.py
```
Access the web dashboard at `http://127.0.0.1:5000`.

---

## 📂 Project Structure

```plaintext
College-Admission-Agent/
│
├── app.py                     # Core Flask server and LangChain RAG orchestration
├── app.json                   # Project metadata and configuration
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variable template
├── university_data.txt        # The unstructured institutional knowledge base
├── Problem_Statement.pdf      # Official AICTE challenge document
└── Project_Presentation.pptx  # Architecture and solution overview slides
```
