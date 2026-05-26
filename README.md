# 📄 Autonomous RFP & Tender Agent

An Enterprise-grade Retrieval-Augmented Generation (RAG) system built to automate business proposals, Requests for Proposals (RFPs), and government tenders. 

This AI agent uses a strict closed-loop memory to eliminate hallucination, drafting professional responses based **only** on your company's uploaded private documents.

## ✨ Features
* **Zero-Hallucination RAG:** Powered by local vector search (ChromaDB), ensuring the AI only pulls facts from approved company data.
* **Interactive Audit UI:** Users can visually verify the exact database paragraphs the AI retrieved to write its answer.
* **Lightning Fast Generation:** Utilizes Groq's Llama-3.1 API for instant text generation.
* **Document Export:** Built-in one-click download to generate ready-to-send text files.

## 🛠️ Tech Stack
* **Frontend:** Streamlit
* **AI Orchestration:** LangChain
* **LLM:** Meta Llama 3.1 (via Groq)
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
* **Vector Database:** ChromaDB

## 🚀 How to Run Locally

**1. Clone the repository**
```bash
git clone [https://github.com/pritansh07/autonomous-rfp-agent.git](https://github.com/pritansh07/autonomous-rfp-agent.git)
cd autonomous-rfp-agent
```

**2. Create a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your API Key**
Create a `.env` file in the root directory and add your Groq API key:
```text
GROQ_API_KEY=gsk_your_key_here
```

**5. Build the Database**
Add your company PDFs to the `data/` folder, then run the ingestion script:
```bash
python3 ingest.py
```

**6. Launch the App**
```bash
streamlit run app.py
```
