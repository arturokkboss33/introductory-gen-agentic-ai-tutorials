import os
import streamlit as st
import pandas as pd
import shutil
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from umap import UMAP

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
        # Use the stored database path if available, otherwise use default
        if "db_path" in st.session_state:
            db_dir = st.session_state.db_path
        else:
            # Create a fresh database directory with timestamp
            import time
            timestamp = int(time.time())
            db_dir = f'./knowledge_db_{timestamp}'
            st.session_state.db_path = db_dir
        
        # Ensure the database directory exists and is writable
        os.makedirs(db_dir, exist_ok=True)
        
        # Set proper permissions
        try:
            import stat
            os.chmod(db_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        except:
            pass  # Ignore permission errors on some systems
        
        # Create a unique collection name to avoid conflicts
        import time
        collection_name = f"knowledge_database_{int(time.time())}"
        
        try:
            st.session_state.db = Chroma(
                collection_name=collection_name,
                embedding_function=st.session_state.embedding_model,
                persist_directory=db_dir
            )
            st.info(f"✅ Database initialized at: {db_dir}")
        except Exception as e:
            # If database creation fails, try with a completely fresh directory
            st.warning(f"Database creation failed: {e}. Trying with fresh directory...")
            
            # Create a new timestamped directory
            fresh_timestamp = int(time.time()) + 1
            fresh_db_dir = f'./knowledge_db_{fresh_timestamp}_fresh'
            st.session_state.db_path = fresh_db_dir
            
            os.makedirs(fresh_db_dir, exist_ok=True)
            
            # Try again with a new collection name
            collection_name = f"knowledge_database_{fresh_timestamp}_fresh"
            st.session_state.db = Chroma(
                collection_name=collection_name,
                embedding_function=st.session_state.embedding_model,
                persist_directory=fresh_db_dir
            )
            st.success(f"✅ Fresh database created at: {fresh_db_dir}")

def format_docs(docs):
    """Formats a list of document objects into a single string.

    Args:
        docs (list): A list of document objects, each having a 'page_content' attribute.

    Returns:
        str: A single string containing the page content from each document, 
        separated by double newlines."""
    return "\n\n".join(doc.page_content for doc in docs)

def reset_database():
    """Reset the database by removing all documents and clearing session state."""
    try:
        # Clear session state WITHOUT trying to interact with the corrupted database
        session_keys_to_clear = [
            "db", "embedding_model", "document_info", "full_embeddings_data",
            "chunking_params_set", "current_chunk_size", "current_chunk_overlap"
        ]
        
        for key in session_keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        # Import time for delays
        import time
        time.sleep(1)
        
        # Generate a new database directory name to avoid any corruption
        timestamp = int(time.time())
        new_db_path = f'./knowledge_db_{timestamp}'
        old_db_path = './knowledge_db'
        
        # Store the new database path for future use
        st.session_state.db_path = new_db_path
        
        # Try to handle the old database directory
        if os.path.exists(old_db_path):
            try:
                # Don't try to delete - just rename it out of the way
                backup_name = f'./knowledge_db_old_{timestamp}'
                os.rename(old_db_path, backup_name)
                st.info(f"Old database moved to {backup_name}")
            except Exception as e:
                st.warning(f"Could not move old database: {e}")
        
        # Remove temp directory if it exists
        temp_path = './temp'
        if os.path.exists(temp_path):
            try:
                shutil.rmtree(temp_path)
            except Exception as e:
                st.warning(f"Could not remove temp directory: {str(e)}")
        
        # Force garbage collection
        import gc
        gc.collect()
            
        st.success(f"✅ Database reset successfully! New database will be created at: {new_db_path}")
        return True
        
    except Exception as e:
        st.error(f"Error resetting database: {str(e)}")
        return False

def add_to_db(uploaded_files, chunk_size, chunk_overlap):
    """Processes and adds uploaded PDF files to the database.

    This function checks if any files have been uploaded. If files are uploaded,
    it saves each file to a temporary location, processes the content using a PDF loader,
    and splits the content into smaller chunks using the specified parameters. Each chunk, 
    along with its metadata, is then added to the database. Temporary files are removed after processing.

    Args:
        uploaded_files (list): A list of uploaded file objects to be processed.
        chunk_size (int): The size of each chunk in tokens.
        chunk_overlap (int): The number of tokens to overlap between chunks.

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
    
    # Initialize full embeddings storage for visualization
    if "full_embeddings_data" not in st.session_state:
        st.session_state.full_embeddings_data = []

    # Store chunking parameters for consistency
    st.session_state.chunking_params_set = True
    st.session_state.current_chunk_size = chunk_size
    st.session_state.current_chunk_overlap = chunk_overlap

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

        # Split documents into smaller chunks with user-specified parameters
        st_text_splitter = SentenceTransformersTokenTextSplitter(
            model_name="sentence-transformers/all-mpnet-base-v2",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        st_chunks = st_text_splitter.create_documents(doc_content, doc_metadata)

        # Add chunks to database
        st.session_state.db.add_documents(st_chunks)

        # Generate embeddings for ALL chunks (for visualization)
        for i, chunk in enumerate(st_chunks):
            # Get embedding for this chunk
            embedding = st.session_state.embedding_model.embed_query(chunk.page_content)
            
            # Store full embedding data for visualization
            embedding_data = {
                "document_name": uploaded_file.name,
                "chunk_index": i,
                "chunk_text": chunk.page_content,
                "embedding": embedding,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap
            }
            st.session_state.full_embeddings_data.append(embedding_data)

        # Generate embeddings for display table (limit to 5 chunks per document)
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
                "Embedding Length": len(embedding),
                "Chunk Size": chunk_size,
                "Chunk Overlap": chunk_overlap
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

def subsample_chunks_per_document(embeddings_data, max_chunks_per_doc):
    """Subsample chunks per document to the specified maximum number.
    
    Args:
        embeddings_data (list): List of embedding data dictionaries
        max_chunks_per_doc (int): Maximum number of chunks to keep per document
        
    Returns:
        list: Subsampled embedding data
    """
    # Group data by document name
    doc_groups = {}
    for item in embeddings_data:
        doc_name = item['document_name']
        if doc_name not in doc_groups:
            doc_groups[doc_name] = []
        doc_groups[doc_name].append(item)
    
    # Subsample each document
    subsampled_data = []
    for doc_name, chunks in doc_groups.items():
        if len(chunks) <= max_chunks_per_doc:
            # If document has fewer chunks than max, keep all
            subsampled_data.extend(chunks)
        else:
            # Uniformly subsample chunks
            indices = np.linspace(0, len(chunks) - 1, max_chunks_per_doc, dtype=int)
            subsampled_chunks = [chunks[i] for i in indices]
            subsampled_data.extend(subsampled_chunks)
    
    return subsampled_data

def create_embedding_visualization(visualization_type, dimensions, max_chunks_per_doc):
    """Create UMAP or t-SNE visualization of embeddings.
    
    Args:
        visualization_type (str): Either 'UMAP' or 't-SNE'
        dimensions (str): Either '2D' or '3D'
        max_chunks_per_doc (int): Maximum number of chunks per document to visualize
        
    Returns:
        tuple: (plotly.graph_objects.Figure, dict) - The visualization plot and stats
    """
    try:
        # Check if embedding data exists
        if "full_embeddings_data" not in st.session_state:
            st.warning("❌ No embedding data found in session state. Please upload and process documents first.")
            return None, None
            
        if not st.session_state.full_embeddings_data:
            st.warning("❌ Embedding data is empty. Please upload and process documents first.")
            return None, None
        
        st.info(f"📊 Starting {visualization_type} {dimensions} computation...")
        st.info(f"📈 Total available chunks: {len(st.session_state.full_embeddings_data)}")
        
        # Subsample data if needed
        data_to_visualize = subsample_chunks_per_document(
            st.session_state.full_embeddings_data, 
            max_chunks_per_doc
        )
        
        if len(data_to_visualize) == 0:
            st.warning("❌ No data to visualize after subsampling.")
            return None, None
        
        st.info(f"📈 Processing {len(data_to_visualize)} chunks from {len(set([item['document_name'] for item in data_to_visualize]))} documents...")
        
        # Extract embeddings and document names
        try:
            st.info("🔄 Converting embeddings to numpy array...")
            
            # Extract embeddings and convert to proper numpy array
            embedding_list = []
            for item in data_to_visualize:
                embedding = item['embedding']
                # Convert to numpy array if it's a list
                if isinstance(embedding, list):
                    embedding = np.array(embedding)
                elif not isinstance(embedding, np.ndarray):
                    embedding = np.array(embedding)
                embedding_list.append(embedding)
            
            embeddings = np.vstack(embedding_list)
            
            document_names = [item['document_name'] for item in data_to_visualize]
            chunk_texts = [item['chunk_text'][:100] + "..." if len(item['chunk_text']) > 100 
                           else item['chunk_text'] for item in data_to_visualize]
                           
            st.info(f"✅ Successfully converted embeddings. Shape: {embeddings.shape}")
            
        except Exception as e:
            st.error(f"❌ Error extracting/converting embeddings: {str(e)}")
            st.error(f"Error details: {type(e).__name__}")
            
            # Debug information
            st.write("**Debug - First few embedding items:**")
            for i, item in enumerate(data_to_visualize[:3]):
                embedding = item['embedding']
                st.write(f"Item {i}: type={type(embedding)}, length={len(embedding) if hasattr(embedding, '__len__') else 'N/A'}")
                if hasattr(embedding, '__len__') and len(embedding) > 0:
                    st.write(f"  First few values: {embedding[:5] if isinstance(embedding, (list, np.ndarray)) else 'Cannot display'}")
            
            return None, None
        
        st.info(f"🔢 Embedding matrix shape: {embeddings.shape}")
        
        # Check for valid embeddings
        if embeddings.size == 0:
            st.error("❌ Embeddings array is empty")
            return None, None
            
        if len(embeddings.shape) != 2:
            st.error(f"❌ Invalid embedding shape: {embeddings.shape}. Expected 2D array.")
            return None, None
        
        # Determine number of components
        n_components = 2 if dimensions == '2D' else 3
        
        # Check minimum data requirements
        min_samples_needed = max(n_components + 1, 4)
        if len(embeddings) < min_samples_needed:
            st.error(f"❌ Need at least {min_samples_needed} data points for {visualization_type} {dimensions}. Found {len(embeddings)} points.")
            return None, None
        
        # Apply dimensionality reduction
        st.info(f"🧮 Computing {visualization_type} with {n_components} components...")
        
        try:
            if visualization_type == 'UMAP':
                # UMAP parameters adjusted for small datasets
                n_neighbors = min(15, len(embeddings) - 1)
                if n_neighbors < 2:
                    n_neighbors = 2
                reducer = UMAP(
                    n_components=n_components, 
                    random_state=42,
                    n_neighbors=n_neighbors,
                    min_dist=0.1,
                    metric='cosine'
                )
                reduced_embeddings = reducer.fit_transform(embeddings)
            else:  # t-SNE
                # t-SNE parameters adjusted for small datasets
                perplexity = min(30, max(5, len(embeddings) - 1))
                if perplexity < 1:
                    perplexity = 1
                reducer = TSNE(
                    n_components=n_components, 
                    random_state=42, 
                    perplexity=perplexity,
                    max_iter=1000,
                    learning_rate='auto'
                )
                reduced_embeddings = reducer.fit_transform(embeddings)
        except Exception as e:
            st.error(f"❌ Error during {visualization_type} computation: {str(e)}")
            return None, None
        
        st.info(f"✅ Dimensionality reduction complete! Shape: {reduced_embeddings.shape}")
        
        # Create DataFrame for plotting
        try:
            plot_data = {
                'x': reduced_embeddings[:, 0],
                'y': reduced_embeddings[:, 1],
                'document': document_names,
                'chunk_text': chunk_texts
            }
            
            if n_components == 3:
                plot_data['z'] = reduced_embeddings[:, 2]
            
            df_plot = pd.DataFrame(plot_data)
        except Exception as e:
            st.error(f"❌ Error creating plot data: {str(e)}")
            return None, None
        
        st.info("📊 Creating interactive plot...")
        
        # Create plot
        try:
            if dimensions == '2D':
                fig = px.scatter(
                    df_plot, 
                    x='x', 
                    y='y', 
                    color='document',
                    hover_data=['chunk_text'],
                    title=f'{visualization_type} {dimensions} Visualization of Document Embeddings',
                    labels={'x': f'{visualization_type} 1', 'y': f'{visualization_type} 2'},
                    width=800,
                    height=600
                )
            else:  # 3D
                fig = px.scatter_3d(
                    df_plot, 
                    x='x', 
                    y='y', 
                    z='z',
                    color='document',
                    hover_data=['chunk_text'],
                    title=f'{visualization_type} {dimensions} Visualization of Document Embeddings',
                    labels={'x': f'{visualization_type} 1', 'y': f'{visualization_type} 2', 'z': f'{visualization_type} 3'},
                    width=800,
                    height=600
                )
        except Exception as e:
            st.error(f"❌ Error creating plot: {str(e)}")
            return None, None
        
        # Update layout
        try:
            fig.update_layout(
                legend_title="Documents",
                showlegend=True,
                font=dict(size=12),
                margin=dict(l=20, r=20, t=60, b=20)
            )
            
            # Update hover template
            if dimensions == '2D':
                hovertemplate = ('<b>Document:</b> %{customdata[0]}<br>' +
                               '<b>Chunk Text:</b> %{customdata[1]}<br>' +
                               '<b>Coordinates:</b> (%{x:.2f}, %{y:.2f})<br>' +
                               '<extra></extra>')
            else:
                hovertemplate = ('<b>Document:</b> %{customdata[0]}<br>' +
                               '<b>Chunk Text:</b> %{customdata[1]}<br>' +
                               '<b>Coordinates:</b> (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>' +
                               '<extra></extra>')
            
            fig.update_traces(
                hovertemplate=hovertemplate,
                customdata=list(zip(document_names, chunk_texts))
            )
        except Exception as e:
            st.error(f"❌ Error updating plot layout: {str(e)}")
            return None, None
        
        # Calculate statistics
        try:
            doc_chunk_counts = {}
            for item in data_to_visualize:
                doc_name = item['document_name']
                doc_chunk_counts[doc_name] = doc_chunk_counts.get(doc_name, 0) + 1
            
            stats = {
                'total_points': len(data_to_visualize),
                'doc_counts': doc_chunk_counts,
                'embedding_dim': embeddings.shape[1],
                'reduced_dim': n_components
            }
        except Exception as e:
            st.error(f"❌ Error calculating statistics: {str(e)}")
            return None, None
        
        st.success("🎉 Visualization created successfully!")
        
        return fig, stats
        
    except Exception as e:
        st.error(f"❌ Unexpected error in visualization: {str(e)}")
        st.error(f"Error type: {type(e).__name__}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None, None

def clear_models():
    """Clear cached models from session state when API key changes."""
    if "embedding_model" in st.session_state:
        del st.session_state.embedding_model
    if "db" in st.session_state:
        del st.session_state.db
    if "document_info" in st.session_state:
        del st.session_state.document_info
    if "full_embeddings_data" in st.session_state:
        del st.session_state.full_embeddings_data

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
        st.subheader("📊 Document Processing Settings")
        
        # Show current chunking parameters if they exist
        if st.session_state.get("chunking_params_set"):
            st.info(f"🔧 **Current Settings:**\n- Chunk Size: {st.session_state.get('current_chunk_size', 'N/A')}\n- Chunk Overlap: {st.session_state.get('current_chunk_overlap', 'N/A')}")
            st.warning("⚠️ Chunking parameters are locked after uploading files. Use 'Reset Database' to change them.")
        
        # Chunking parameters (disabled if already set)
        chunk_size = st.number_input(
            "Chunk Size (tokens):",
            min_value=50,
            max_value=1000,
            value=st.session_state.get('current_chunk_size', 100),
            step=10,
            help="Number of tokens per chunk",
            disabled=st.session_state.get("chunking_params_set", False)
        )
        
        chunk_overlap = st.number_input(
            "Chunk Overlap (tokens):",
            min_value=0,
            max_value=min(chunk_size-1, 200),
            value=st.session_state.get('current_chunk_overlap', 50),
            step=5,
            help="Number of overlapping tokens between chunks",
            disabled=st.session_state.get("chunking_params_set", False)
        )
        
        # Validation
        if chunk_overlap >= chunk_size:
            st.error("⚠️ Chunk overlap must be less than chunk size!")
        
        # Reset Database button
        if st.button("🗑️ Reset Database", type="secondary"):
            if reset_database():
                st.success("✅ Database reset successfully! You can now upload new documents with different chunking parameters.")
                st.rerun()
            else:
                st.error("❌ Failed to reset database. Please try again.")
        
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
            elif chunk_overlap >= chunk_size:
                st.error("Please fix the chunking parameters: overlap must be less than chunk size!")
            else:
                with st.spinner("Processing your documents..."):
                    add_to_db(pdf_docs, chunk_size, chunk_overlap)
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
                "Embedding Length": st.column_config.NumberColumn("Vector Dimensions", width="small"),
                "Chunk Size": st.column_config.NumberColumn("Chunk Size", width="small"),
                "Chunk Overlap": st.column_config.NumberColumn("Chunk Overlap", width="small")
            }
        )
        
        st.caption(f"📝 Showing up to 5 chunks per document. Total chunks displayed: {len(df)}")
        
        # Add button to clear document info
        if st.button("🗑️ Clear Document Table"):
            st.session_state.document_info = []
            st.rerun()

    # Embedding Visualization Section
    if "full_embeddings_data" in st.session_state and st.session_state.full_embeddings_data:
        st.subheader("🎯 Embedding Visualization")
        
        # Get document count for reference
        doc_names = list(set([item['document_name'] for item in st.session_state.full_embeddings_data]))
        total_chunks = len(st.session_state.full_embeddings_data)
        
        st.info(f"📚 **Available Data:** {len(doc_names)} documents, {total_chunks} total chunks")
        
        # Debug information
        with st.expander("🔍 Debug Information", expanded=False):
            st.write("**Sample embedding data structure:**")
            if st.session_state.full_embeddings_data:
                sample_item = st.session_state.full_embeddings_data[0]
                st.json({
                    "document_name": sample_item.get('document_name', 'N/A'),
                    "chunk_index": sample_item.get('chunk_index', 'N/A'),
                    "embedding_length": len(sample_item.get('embedding', [])) if sample_item.get('embedding') else 0,
                    "embedding_type": type(sample_item.get('embedding', None)).__name__,
                    "chunk_text_length": len(sample_item.get('chunk_text', ''))
                })
                
                st.write("**Document distribution:**")
                doc_counts = {}
                for item in st.session_state.full_embeddings_data:
                    doc_name = item.get('document_name', 'Unknown')
                    doc_counts[doc_name] = doc_counts.get(doc_name, 0) + 1
                st.write(doc_counts)
        
        # Visualization parameters in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            viz_type = st.selectbox(
                "Visualization Method:",
                ["UMAP", "t-SNE"],
                help="UMAP is generally faster and better for preserving global structure"
            )
        
        with col2:
            dimensions = st.selectbox(
                "Plot Dimensions:",
                ["2D", "3D"],
                help="3D plots provide more detailed view but can be harder to interpret"
            )
        
        with col3:
            max_chunks = st.number_input(
                "Max chunks per document:",
                min_value=10,
                max_value=500,
                value=50,
                step=10,
                help="Higher values give more detail but slower computation"
            )
        
        # Generate visualization button
        if st.button("🎨 Generate Visualization", type="primary"):
            # Create progress container
            progress_container = st.container()
            plot_container = st.container()
            
            with progress_container:
                # Show initial status
                status_placeholder = st.empty()
                progress_bar = st.progress(0)
                
                try:
                    # Update progress
                    status_placeholder.info("🚀 Initializing visualization...")
                    progress_bar.progress(10)
                    
                    # Generate the visualization
                    with st.spinner(f"Computing {viz_type} {dimensions} visualization..."):
                        progress_bar.progress(30)
                        status_placeholder.info(f"🧮 Running {viz_type} algorithm...")
                        
                        result = create_embedding_visualization(viz_type, dimensions, max_chunks)
                        
                        # Check if result is valid
                        if result is None:
                            progress_bar.progress(100)
                            status_placeholder.error("❌ Visualization failed - no result returned")
                            st.error("The visualization function returned no result. Please check your data and try again.")
                        elif result == (None, None):
                            progress_bar.progress(100)
                            status_placeholder.error("❌ Visualization failed - check error messages above")
                        else:
                            # Unpack the result safely
                            fig, stats = result
                            
                            if fig is None or stats is None:
                                progress_bar.progress(100)
                                status_placeholder.error("❌ Visualization partially failed - invalid results")
                            else:
                                progress_bar.progress(80)
                                status_placeholder.info("📊 Rendering interactive plot...")
                                
                                progress_bar.progress(100)
                                status_placeholder.success("✅ Visualization completed successfully!")
                                
                                # Clear progress indicators after a short delay
                                import time
                                time.sleep(1)
                                progress_container.empty()
                                
                                with plot_container:
                                    # Display the plot
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Show statistics about the visualization
                                    st.caption("📊 **Visualization Statistics:**")
                                    
                                    # Create statistics dataframe
                                    stats_df = pd.DataFrame([
                                        {"Document": doc, "Chunks Visualized": count} 
                                        for doc, count in stats['doc_counts'].items()
                                    ])
                                    
                                    col_stats1, col_stats2 = st.columns(2)
                                    with col_stats1:
                                        st.dataframe(stats_df, hide_index=True, use_container_width=True)
                                    
                                    with col_stats2:
                                        st.metric("Total Points", stats['total_points'])
                                        st.metric("Original Dimensions", stats['embedding_dim'])
                                        st.metric("Reduced Dimensions", stats['reduced_dim'])
                        
                except Exception as e:
                    progress_bar.progress(100)
                    status_placeholder.error(f"❌ Unexpected error: {str(e)}")
                    st.error(f"Error details: {type(e).__name__}")
                    import traceback
                    with st.expander("🔍 Full Error Details"):
                        st.code(traceback.format_exc())

    # Sidebar Footer
    st.sidebar.write("Arturo Gomez-Chavez")
             
if __name__ == "__main__":
    main()