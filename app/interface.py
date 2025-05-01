# Streamlit App

import streamlit as st
from app.rag_chain import main  


st.set_page_config(page_title="Pinecone + Groq QA", layout="centered")
st.title("🧠 Customer Support Assistant")

query = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if query.strip():
        with st.spinner("Thinking..."):
            answer = main(query)
        st.success("Answer:")
        st.write(answer)
    else:
        st.warning("Please enter a valid question.")
