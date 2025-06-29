import os
import streamlit as st
import pandas as pd

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.sentence_transformers import SentenceTransformersTokenTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def initialize_models():
    """Initialize embedding model and database only once using session state cache."""
    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=st.session_state.get("gemini_api_key")
        )
    
    if "db" not in st.session_state:
        st.session_state.db = Chroma(
            collection_name="knowledge_database",
            embedding_function=st.session_state.embedding_model,
            persist_directory='./knowledge_db'
        )

def format_docs(docs):
    """Formats a list of document objects into a single string.

    Args:
        docs (list): A list of document objects, each having a 'page_content' attribute.

    Returns:
        str: A single string containing the page content from each document, 
        separated by double newlines."""
    return "\n\n".join(doc.page_content for doc in docs)

def add_to_db(uploaded_files):
    """Processes and adds uploaded PDF files to the database.

    This function checks if any files have been uploaded. If files are uploaded,
    it saves each file to a temporary location, processes the content using a PDF loader,
    and splits the content into smaller chunks. Each chunk, along with its metadata, 
    is then added to the database. Temporary files are removed after processing.

    Args:
        uploaded_files (list): A list of uploaded file objects to be processed.

    Returns:
        None"""
    # Check if files are uploaded
    if not uploaded_files:
        st.error("No files uploaded!")
        return

    # Check if API key is available
    if not st.session_state.get("gemini_api_key"):
        st.error("Please enter your Gemini API key first!")
        return

    # Initialize models if not already done
    initialize_models()

    # Initialize document info list if not exists
    if "document_info" not in st.session_state:
        st.session_state.document_info = []

    for uploaded_file in uploaded_files:
        # Save the uploaded file to a temporary path
        temp_file_path = os.path.join("./temp", uploaded_file.name)
        os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)

        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.getbuffer())

        # Load the file using PyPDFLoader
        loader = PyPDFLoader(temp_file_path)
        data = loader.load()

        # Store metadata and content
        doc_metadata = [data[i].metadata for i in range(len(data))]
        doc_content = [data[i].page_content for i in range(len(data))]

        # Split documents into smaller chunks
        st_text_splitter = SentenceTransformersTokenTextSplitter(
            model_name="sentence-transformers/all-mpnet-base-v2",
            chunk_size=100,
            chunk_overlap=50
        )
        st_chunks = st_text_splitter.create_documents(doc_content, doc_metadata)

        # Add chunks to database
        st.session_state.db.add_documents(st_chunks)

        # Generate embeddings for display (limit to 5 chunks per document)
        chunks_to_display = st_chunks[:5]
        
        for i, chunk in enumerate(chunks_to_display):
            # Get embedding for this chunk
            embedding = st.session_state.embedding_model.embed_query(chunk.page_content)
            
            # Store document info for display
            doc_info = {
                "Document Name": uploaded_file.name,
                "Chunk Index": i,
                "Chunk Text": chunk.page_content[:100] + "..." if len(chunk.page_content) > 100 else chunk.page_content,
                "Embedding": str(embedding[:10]) + "..." if len(embedding) > 10 else str(embedding),  # Show first 10 dimensions
                "Embedding Length": len(embedding)
            }
            st.session_state.document_info.append(doc_info)

        # Remove the temporary file after processing
        os.remove(temp_file_path)

def run_rag_chain(query, custom_prompt=None):
    """Processes a query using a Retrieval-Augmented Generation (RAG) chain.

    This function utilizes a RAG chain to answer a given query. It retrieves 
    relevant context using similarity search and then generates a response 
    based on this context using a chat model. The chat model is pre-configured 
    with a prompt template specialized in  sciences.

    Args:
        query (str): The user's question that needs to be answered.
        custom_prompt (str, optional): Custom prompt template. If None, uses default.

    Returns:
        str: A response generated by the chat model, based on the retrieved context."""
    
    # Check if API key is available
    if not st.session_state.get("gemini_api_key"):
        st.error("Please enter your Gemini API key first!")
        return "API key required to process queries."

    # Initialize models if not already done
    initialize_models()
    
    # Create a Retriever Object and apply Similarity Search
    retriever = st.session_state.db.as_retriever(search_type="similarity", search_kwargs={'k': 5})

    # Use custom prompt if provided, otherwise use default
    if custom_prompt:
        PROMPT_TEMPLATE = custom_prompt
    else:
        PROMPT_TEMPLATE = """You are a highly knowledgeable assistant specializing in Environmental sciences. 
Answer the question based only on the following context:
{context}

Answer the question based on the above context:
{question}

Use the provided context to answer the user's question accurately and concisely.
Don't justify your answers.
Don't give information not mentioned in the CONTEXT INFORMATION.
Do not say "according to the context" or "mentioned in the context" or similar."""

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    # Initialize a Generator (i.e. Chat Model)
    chat_model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        api_key=st.session_state.get("gemini_api_key"),
        temperature=1
    )

    # Initialize a Output Parser
    output_parser = StrOutputParser()

    # RAG Chain
    rag_chain = {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt_template | chat_model | output_parser

    # Invoke the Chain
    response = rag_chain.invoke(query)

    return response

def clear_models():
    """Clear cached models from session state when API key changes."""
    if "embedding_model" in st.session_state:
        del st.session_state.embedding_model
    if "db" in st.session_state:
        del st.session_state.db
    if "document_info" in st.session_state:
        del st.session_state.document_info

def main():
    """Initialize and manage the KnowledgeQuery application interface.

    This function sets up the Streamlit application interface for KnowledgeQuery,
    a  Insight Retrieval System. Users can enter queries related
    to the  industry, upload research documents, and manage API 
    keys for enhanced functionality.

    The main features include:
    - Query input area for users to ask questions about the  industry.
    - Submission button to process the query and display the retrieved insights.
    - Sidebar for API key input and management.
    - File uploader for adding research documents to the database, enhancing query responses.

    Args:
        None

    Returns:
        None"""
    st.set_page_config(page_title="knowledgeQuery", page_icon=":microscope:")
    st.header(" Insight Retrieval System")

    # Default prompt template
    default_prompt = """You are a highly knowledgeable assistant specializing in Environmental sciences. 
Answer the question based only on the following context:
{context}

Answer the question based on the above context:
{question}

Use the provided context to answer the user's question accurately and concisely.
Don't justify your answers.
Don't give information not mentioned in the CONTEXT INFORMATION.
Do not say "according to the context" or "mentioned in the context" or similar."""

    # Prompt customization section
    st.subheader("🎯 Customize Assistant Behavior")
    with st.expander("Advanced: Custom Prompt Template", expanded=False):
        st.info("💡 **Important**: Your prompt template must include `{context}` and `{question}` placeholders for the system to work properly.")
        
        custom_prompt = st.text_area(
            "Enter your custom prompt template:",
            value=default_prompt,
            height=200,
            help="Modify this template to change how the assistant responds. Keep {context} and {question} placeholders."
        )
        
        if st.button("Reset to Default"):
            st.rerun()

    query = st.text_area(
        ":bulb: Enter your query about the  Industry:",
        placeholder="e.g., What are the AI applications in drug discovery?"
    )

    if st.button("Submit"):
        if not query:
            st.warning("Please ask a question")
        elif not st.session_state.get("gemini_api_key"):
            st.warning("Please enter your Gemini API key first!")
        else:
            # Validate that custom prompt has required placeholders
            if "{context}" not in custom_prompt or "{question}" not in custom_prompt:
                st.error("⚠️ Your custom prompt template must include both `{context}` and `{question}` placeholders!")
            else:
                with st.spinner("Thinking..."):
                    result = run_rag_chain(query=query, custom_prompt=custom_prompt)
                    st.write(result)

    with st.sidebar:
        st.title("API Keys")
        gemini_api_key = st.text_input("Enter your Gemini API key:", type="password")

        if st.button("Enter"):
            if gemini_api_key:
                # Check if API key has changed, if so clear cached models
                if st.session_state.get("gemini_api_key") != gemini_api_key:
                    clear_models()
                
                st.session_state.gemini_api_key = gemini_api_key
                st.success("API key saved!")

            else:
                st.warning("Please enter your Gemini API key to proceed.")
    
    with st.sidebar:
        st.markdown("---")
        pdf_docs = st.file_uploader("Upload your research documents related to  Sciences (Optional) :memo:",
                                    type=["pdf"],
                                    accept_multiple_files=True
        )
        
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload the file")
            elif not st.session_state.get("gemini_api_key"):
                st.warning("Please enter your Gemini API key first!")
            else:
                with st.spinner("Processing your documents..."):
                    add_to_db(pdf_docs)
                    st.success(":file_folder: Documents successfully added to the database!")

    # Document Information Table
    if "document_info" in st.session_state and st.session_state.document_info:
        st.subheader("📊 Document Chunks & Embeddings")
        
        # Convert to DataFrame for better display
        df = pd.DataFrame(st.session_state.document_info)
        
        # Display the table
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Document Name": st.column_config.TextColumn("Document", width="medium"),
                "Chunk Index": st.column_config.NumberColumn("Chunk #", width="small"),
                "Chunk Text": st.column_config.TextColumn("Content Preview", width="large"),
                "Embedding": st.column_config.TextColumn("Vector Embedding (first 10 dims)", width="large"),
                "Embedding Length": st.column_config.NumberColumn("Vector Dimensions", width="small")
            }
        )
        
        st.caption(f"📝 Showing up to 5 chunks per document. Total chunks displayed: {len(df)}")
        
        # Add button to clear document info
        if st.button("🗑️ Clear Document Table"):
            st.session_state.document_info = []
            st.rerun()

    # Sidebar Footer
    st.sidebar.write("Arturo Gomez-Chavez")
             
if __name__ == "__main__":
    main()