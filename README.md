# 🤖 RAG Chatbot for Customer Support

A Retrieval-Augmented Generation (RAG) chatbot that enhances customer support by retrieving relevant documentation and generating accurate responses using **Groq's LLaMA-3.1-8B** model. The chatbot features an intuitive **Streamlit UI** and responds strictly based on the provided context.

> 💡 For queries outside the provided documentation, it replies: **"I don't know."**

---

## 🚀 Key Features

- 🔍 **Smart Document Retrieval**  
  Uses **Pinecone** to fetch the top 3 relevant documents.

- 🤖 **Context-Aware Answer Generation**  
  Powered by **Groq's LLaMA-3.1-8B** for precise responses.

- ❓ **Out-of-Scope Detection**  
  Answers _"I don't know"_ if information isn't available in the documents.

- 🖥️ **Interactive UI**  
  Built with **Streamlit**, includes expandable context sections.

- 🧱 **Modular Codebase**  
  Clean separation between UI (`interface.py`) and backend logic (`rag_chain.py`).

---

## 📁 Project Structure

```
CUSTOMER-SUPPORT-RAG/
├── app/
│   ├── __pycache__/
│   ├── documents.json              # Indexed document data
│   ├── ingest.ipynb                # Notebook for data ingestion
│   ├── interface.py                # Streamlit app
│   └── rag_chain.py                # Core RAG logic
├── 01_data_gathering_logic/
│   ├── angelone_quick_10_links_support_data.json
│   ├── angelone_support_full_data.json
│   ├── insurance_pdfs_flat.json
│   └── Insurance PDFs/
│       └── PDFs/
├── .env                            # Environment variables (GROQ_API_KEY & PINECONE_API_KEY)
├── README.md
├── requirements.txt
```

---

## 🛠️ Installation

Install all dependencies:

```bash
pip install -r requirements.txt
```

### Included in `requirements.txt`:

```txt
langchain==0.1.10
openai==1.19.0
pinecone
sentence-transformers==2.2.2
PyPDF2==3.0.1
pdfplumber==0.10.2
beautifulsoup4==4.12.3
requests==2.31.0
streamlit==1.33.0
tqdm==4.66.2
numpy==1.26.4
pandas==2.2.2
python-dotenv==1.0.1
```

---

## 🔑 Environment Configuration

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

---

## 🚦 How to Run

### 1. Ingest Data

Run the ingestion notebook to generate the document index:

```bash
open app/ingest.ipynb
```

- Output: `documents.json`

### 2. Launch the Chatbot

```bash
streamlit run app/interface.py
```

### 3. Start Interacting

- Open: [http://localhost:8501](http://localhost:8501)
- Ask questions based on the uploaded support documents.
- View supporting documents by expanding the UI panels.

---

## 🎯 Example Chatbot Behavior

### ✅ Supported Question

- When the answer is found in the documents:

![In-context response](image.png)

### ❌ Unsupported Question

- When the query is not covered:

![Out-of-context response](image-1.png)

---

## 📝 Notes

- Responses are limited to the content in `documents.json`.
- Ensure `documents.json` remains in the `/app` directory.

---

## ❗ Troubleshooting

- **Missing Document Index**  
  → Re-run `ingest.ipynb`.

- **API Key Not Found**  
  → Verify `.env` file exists and contains valid keys.

- **Package Errors**  
  → Confirm all packages are installed with the correct versions.

---

## 📚 Resources

- [LangChain Documentation](https://docs.langchain.com/)
- [Pinecone Docs](https://docs.pinecone.io/guides/get-started/overview)
- [Streamlit](https://streamlit.io/)
- [Groq API](https://groq.com/)
