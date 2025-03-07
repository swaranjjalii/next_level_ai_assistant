import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import streamlit as st
from dotenv import load_dotenv
from src.nlp_tasks import NLPTasks
from src.retrieval import VectorRetriever
import tempfile
import shutil

# Load environment variables at application startup
load_dotenv()

def main():
    st.title("AI Assistant")
    
    # Initialize the NLP tasks
    if 'nlp' not in st.session_state:
        st.session_state.nlp = NLPTasks()
    
    # Initialize the retriever
    if 'retriever' not in st.session_state:
        st.session_state.retriever = VectorRetriever()
    
    # Sidebar with functionality selection
    with st.sidebar:
        st.header("Functionality")
        task = st.radio(
            "Select a task",
            ["Summarization", "Sentiment Analysis", "Named Entity Recognition", 
             "Question Answering", "Code Generation", "Code Review", "Document Retrieval"]
        )
        
        # Document upload for RAG
        st.header("Document Management")
        uploaded_files = st.file_uploader("Upload documents for context", 
                                          accept_multiple_files=True, 
                                          type=["txt", "csv", "json", "md"])
        
        if uploaded_files:
            if st.button("Process Documents"):
                # Create temporary directory to store uploaded files
                with tempfile.TemporaryDirectory() as temp_dir:
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                    
                    # Ingest documents
                    st.session_state.retriever.ingest_directory(temp_dir)
                
                st.success(f"Processed {len(uploaded_files)} documents!")
        
        # Index save/load
        index_path = st.text_input("Index path for save/load", "index.json")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Save Index"):
                try:
                    st.session_state.retriever.save_index(index_path)
                    st.success("Index saved!")
                except Exception as e:
                    st.error(f"Error saving index: {e}")
        
        with col2:
            if st.button("Load Index"):
                try:
                    st.session_state.retriever.load_index(index_path)
                    st.success("Index loaded!")
                except Exception as e:
                    st.error(f"Error loading index: {e}")
    
    # Main content area
    if task == "Summarization":
        st.header("Text Summarization")
        text = st.text_area("Enter text to summarize:", height=200)
        if text and st.button("Summarize"):
            result = st.session_state.nlp.summarize(text)
            st.write("### Summary")
            st.write(result)
    
    elif task == "Sentiment Analysis":
        st.header("Sentiment Analysis")
        text = st.text_area("Enter text to analyze:", height=150)
        if text and st.button("Analyze Sentiment"):
            result = st.session_state.nlp.sentiment_analysis(text)
            st.write("### Sentiment")
            st.write(result)
    
    elif task == "Named Entity Recognition":
        st.header("Named Entity Recognition")
        text = st.text_area("Enter text to extract entities:", height=150)
        if text and st.button("Extract Entities"):
            result = st.session_state.nlp.named_entity_recognition(text)
            st.write("### Entities")
            st.write(result)
    
    elif task == "Question Answering":
        st.header("Question Answering")
        question = st.text_input("Enter your question:")
        use_context = st.checkbox("Provide specific context")
        
        if use_context:
            context = st.text_area("Context:", height=150)
        else:
            context = None
            
        if question and st.button("Answer"):
            result = st.session_state.nlp.question_answering(question, context)
            st.write("### Answer")
            st.write(result)
    
    elif task == "Code Generation":
        st.header("Code Generation")
        problem = st.text_area("Describe the coding problem:", height=150)
        if problem and st.button("Generate Code"):
            result = st.session_state.nlp.code_generation(problem)
            st.write("### Generated Code")
            st.code(result)
    
    elif task == "Code Review":
        st.header("Code Review")
        code = st.text_area("Enter Python code to review:", height=200)
        if code and st.button("Review Code"):
            result = st.session_state.nlp.code_review(code)
            st.write("### Code Review")
            st.write(result)
    
    elif task == "Document Retrieval":
        st.header("Document Retrieval")
        query = st.text_input("Enter your search query:")
        top_k = st.slider("Number of results", 1, 10, 3)
        
        if query and st.button("Search"):
            results = st.session_state.retriever.search(query, top_k=top_k)
            
            if results:
                st.write(f"### Found {len(results)} relevant documents")
                for i, (doc_id, score) in enumerate(results):
                    text = st.session_state.retriever.get_document_text(doc_id)
                    metadata = st.session_state.retriever.get_document_metadata(doc_id)
                    
                    with st.expander(f"Result {i+1}: {doc_id} (Score: {score:.4f})"):
                        st.write("**Metadata:**")
                        st.json(metadata)
                        st.write("**Content:**")
                        st.write(text[:500] + ("..." if len(text) > 500 else ""))
            else:
                st.write("No documents found matching your query.")

if __name__ == "__main__":
    main()