"""
utils.py - Utility functions for model and database initialization and management

This module contains functions for:
- Initializing embedding models and vector databases
- Clearing cached models from session state
- Managing database connections and paths
"""

import os
import time
import stat
import streamlit as st
import shutil
import gc
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


def initialize_models():
    """
    Initialize embedding model and database only once using session state cache.
    
    This function creates:
    1. A Google Generative AI embedding model for converting text to vectors
    2. A Chroma vector database for storing and retrieving document embeddings
    
    The models are cached in Streamlit's session state to avoid re-initialization
    on every interaction.
    """
    # Initialize the embedding model if not already created
    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",  # Google's embedding model
            google_api_key=st.session_state.get("gemini_api_key")
        )
    
    # Initialize the vector database if not already created
    if "db" not in st.session_state:
        # Use the stored database path if available, otherwise create a new one
        if "db_path" in st.session_state:
            db_dir = st.session_state.db_path
        else:
            # Create a fresh database directory with timestamp to avoid conflicts
            timestamp = int(time.time())
            db_dir = f'./knowledge_db_{timestamp}'
            st.session_state.db_path = db_dir
        
        # Ensure the database directory exists and is writable
        os.makedirs(db_dir, exist_ok=True)
        
        # Set proper permissions (ignore errors on some systems)
        try:
            os.chmod(db_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        except:
            pass  # Ignore permission errors on some systems
        
        # Create a unique collection name to avoid conflicts
        collection_name = f"knowledge_database_{int(time.time())}"
        
        try:
            # Create the Chroma vector database
            st.session_state.db = Chroma(
                collection_name=collection_name,
                embedding_function=st.session_state.embedding_model,
                persist_directory=db_dir
            )
            #st.info(f"✅ Database initialized at: {db_dir}")
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


def clear_models():
    """
    Clear cached models from session state when API key changes.
    
    This function is called when:
    - The user changes their API key
    - The user wants to reset the system
    
    It removes all cached models and data to ensure fresh initialization
    with the new API key.
    """
    session_keys_to_clear = [
        "embedding_model",      # The embedding model instance
        "db",                   # The vector database instance
        "document_info",        # Document metadata for display
        "full_embeddings_data"  # Embedding data for visualization
    ]
    
    for key in session_keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


def reset_database():
    """
    Reset the database by removing all documents and clearing session state.
    
    This function:
    1. Clears all session state variables related to the database
    2. Creates a new database directory path to avoid corruption
    3. Removes temporary files
    4. Forces garbage collection to free memory
    
    Returns:
        bool: True if reset was successful, False otherwise
    """
    try:
        # Clear session state WITHOUT trying to interact with the corrupted database
        session_keys_to_clear = [
            "db", 
            "embedding_model", 
            "document_info", 
            "full_embeddings_data",
            "chunking_params_set", 
            "current_chunk_size", 
            "current_chunk_overlap"
        ]
        
        for key in session_keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        # Small delay to ensure cleanup
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
        
        # Force garbage collection to free memory
        gc.collect()
            
        st.success(f"✅ Database reset successfully! New database will be created at: {new_db_path}")
        return True
        
    except Exception as e:
        st.error(f"Error resetting database: {str(e)}")
        return False


def get_safe_chunking_defaults(requested_chunk_size=200, requested_overlap=50):
    """
    Calculate safe default values for chunking parameters.
    
    This function ensures that:
    - Chunk overlap is always less than chunk size
    - Default values don't cause Streamlit widget errors
    - Values are reasonable for typical use cases
    
    Args:
        requested_chunk_size (int): Desired chunk size
        requested_overlap (int): Desired chunk overlap
        
    Returns:
        tuple: (safe_chunk_size, safe_overlap)
    """
    # Ensure minimum chunk size
    safe_chunk_size = max(50, requested_chunk_size)
    
    # Calculate maximum allowed overlap
    max_allowed_overlap = safe_chunk_size - 1
    
    # Set safe overlap (prefer 25% of chunk size as a good default)
    recommended_overlap = safe_chunk_size // 4
    
    # Choose the safest option
    if requested_overlap < max_allowed_overlap:
        safe_overlap = requested_overlap
    else:
        safe_overlap = min(recommended_overlap, max_allowed_overlap)
    
    # Ensure overlap is not negative
    safe_overlap = max(0, safe_overlap)
    
    return safe_chunk_size, safe_overlap


def validate_chunking_parameters(chunk_size, chunk_overlap):
    """
    Validate chunking parameters and return validation results.
    
    Args:
        chunk_size (int): Size of each chunk in tokens
        chunk_overlap (int): Overlap between consecutive chunks
        
    Returns:
        dict: Validation results with 'is_valid', 'errors', 'warnings', and 'info'
    """
    result = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'info': []
    }
    
    # Check for critical errors
    if chunk_overlap >= chunk_size:
        result['is_valid'] = False
        result['errors'].append(f"Chunk overlap ({chunk_overlap}) must be less than chunk size ({chunk_size})")
    
    if chunk_size < 20:
        result['is_valid'] = False
        result['errors'].append(f"Chunk size ({chunk_size}) is too small. Minimum recommended: 20 tokens")
    
    # Check for warnings
    if chunk_overlap > chunk_size * 0.7:
        result['warnings'].append(f"Very high overlap ratio ({chunk_overlap/chunk_size*100:.1f}%). Consider reducing overlap.")
    
    if chunk_size > 800:
        result['warnings'].append(f"Very large chunk size ({chunk_size}). This might affect retrieval precision.")
    
    # Provide helpful info
    effective_content = chunk_size - chunk_overlap
    result['info'].append(f"Effective new content per chunk: {effective_content} tokens")
    result['info'].append(f"Overlap ratio: {chunk_overlap/chunk_size*100:.1f}%")
    
    if 50 <= effective_content <= 200:
        result['info'].append("✅ Good balance of chunk size and overlap")
    
    return result
    
def format_docs(docs):

    """
    Formats a list of document objects into a single string for use as context.
    
    This is a utility function used in the RAG chain to convert retrieved documents
    into a format that can be inserted into the prompt template.
    
    Args:
        docs (list): A list of document objects, each having a 'page_content' attribute.
    
    Returns:
        str: A single string containing the page content from each document, 
             separated by double newlines for better readability.
    
    Example:
        If docs contain ["Climate change affects...", "Carbon emissions..."],
        returns: "Climate change affects...\n\nCarbon emissions..."
    """
    return "\n\n".join(doc.page_content for doc in docs)