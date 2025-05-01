RAG Chatbot for Customer Support
This project implements a Retrieval-Augmented Generation (RAG) chatbot trained on customer support documentation. It retrieves relevant information using FAISS and generates answers using the Groq language model (LLaMA-3.1-8B). It includes a Streamlit-based interface and is designed to only answer questions based on the provided context, returning "I don't know" for out-of-scope queries.

🚀 Features

🔍 Document Retrieval: Uses FAISS to fetch top 3 most relevant documents from indexed embeddings.
🤖 Answer Generation: Uses Groq's LLaMA-3.1-8B model to generate contextual answers.
❓ Out-of-Scope Handling: Returns "I don't know" when the answer is not in the provided documents.
🖥️ User Interface: Streamlit UI with expandable document views for transparency.
🧱 Modular Design: Separates UI (app.py) and RAG logic (rag_chain.py) for easy maintenance.


📁 Project Structure
CUSTOMER-SUPPORT-RAG/
├── app/
│   ├── __pycache__/
│   ├── documents.json              # JSON file with document texts
│   ├── faiss_index.index           # FAISS index for retrieval
│   ├── ingest.ipynb                # Notebook for ingesting data
│   ├── interface.py                # Streamlit app interface
│   └── rag_chain.py                # RAG logic and pipeline
│
├── data/
│   ├── angelone_quick_10_links_support_data.json
│   ├── angelone_support_full_data.json
│   └── insurance_pdfs_text.json
│
├── Data Gathering/
│   ├── Insurance PDFs/
│   ├── angelone_quick_10_links_support_data.json
│   ├── angelone_support_full_data.json
│   ├── insurance_pdfs_text.json
│   ├── insurance_pdfs_text1.json
│   └── data_gathering.ipynb
│
├── .env                            # Environment variables (GROQ_API_KEY)
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies



🛠️ Requirements
To install the required dependencies, run:
pip install -r requirements.txt

requirements.txt
langchain==0.1.10
openai==1.19.0
faiss-cpu==1.7.4
sentence-transformers==2.2.2
PyPDF2==3.0.1
pdfplumber==0.10.2
beautifulsoup4==4.12.3
requests==2.31.0
lxml==5.2.1
streamlit==1.33.0
tqdm==4.66.2
numpy==1.26.4
pandas==2.2.2
python-dotenv==1.0.1


🔑 Environment Setup
Create a .env file in the root directory with your Groq API key:
GROQ_API_KEY=your_groq_api_key_here


🚀 Running the Application

Ingest Data (if not already done):

Run the ingest.ipynb notebook to process documents and create the FAISS index (faiss_index.index) and documents.json.


Start the Streamlit App:
streamlit run app/interface.py


Interact with the Chatbot:

Open the provided local URL (e.g., http://localhost:8501) in your browser.
Ask questions related to the customer support documentation.
Expand the document sections to view retrieved context.




📝 Notes

The chatbot is designed to only answer based on the provided documents. If a question is out of scope, it will respond with "I don't know".
Ensure the FAISS index and documents.json are present in the app/ directory before running the app.
The data in the data/ and Data Gathering/ directories includes JSON files with customer support data and extracted text from insurance PDFs.


🛠️ Troubleshooting

Missing FAISS Index: Run ingest.ipynb to generate faiss_index.index and documents.json.
API Key Issues: Verify the GROQ_API_KEY in the .env file.
Dependency Errors: Ensure all packages in requirements.txt are installed correctly using the specified versions.


📚 References

LangChain Documentation
FAISS
Streamlit
Groq API

