import atexit
import glob
import os
import shutil

import gradio as gr
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.config import COLLECTION_NAME, DOCS_DIR, EMBEDDING_MODEL_NAME, MILVUS_DB_PATH
from src.data_loader import load_data
from src.embedding_generator import (
    generate_document_embeddings,
    generate_query_embeddings,
)
from src.rag_pipeline import answer_question
from src.vector_store import (
    create_collection_if_not_exists,
    get_milvus_client,
    insert_data,
    search,
)

# Initialize models and clients
embedding_model = HuggingFaceEmbedding(
    model_name=EMBEDDING_MODEL_NAME,
    trust_remote_code=True,
    cache_folder=".hf_cache",
)

milvus_client = get_milvus_client(MILVUS_DB_PATH)


# --- Cleanup Function ---
def cleanup_documents():
    """Remove all files from the documents directory."""
    print("Cleaning up uploaded documents...")
    files = glob.glob(os.path.join(DOCS_DIR, "*"))
    for f in files:
        if os.path.isfile(f):
            os.remove(f)
    # Also drop the Milvus collection to avoid stale state between restarts
    try:
        if milvus_client and milvus_client.has_collection(COLLECTION_NAME):
            milvus_client.drop_collection(COLLECTION_NAME)
            print(f"Dropped collection {COLLECTION_NAME} during cleanup.")
    except Exception as e:
        print(f"Error dropping collection during cleanup: {e}")
    print("Cleanup complete.")


# Register the cleanup function to run on exit
atexit.register(cleanup_documents)


def reset_collection_if_no_docs():
    """Drop existing collection on startup if there are no documents on disk."""
    try:
        os.makedirs(DOCS_DIR, exist_ok=True)
        files = glob.glob(os.path.join(DOCS_DIR, "*"))
        has_docs = any(os.path.isfile(f) for f in files)
        if (
            not has_docs
            and milvus_client
            and milvus_client.has_collection(COLLECTION_NAME)
        ):
            milvus_client.drop_collection(COLLECTION_NAME)
            print(f"No documents found. Dropped existing collection {COLLECTION_NAME}.")
    except Exception as e:
        print(f"Error resetting collection on startup: {e}")


def index_documents(file_list):
    """Index documents from a list of files."""
    if not file_list:
        return "No files to index."

    os.makedirs(DOCS_DIR, exist_ok=True)

    # Move uploaded files to the documents directory
    for file in file_list:
        shutil.copy(file.name, os.path.join(DOCS_DIR, os.path.basename(file.name)))

    docs = load_data(DOCS_DIR)
    documents = [doc.text for doc in docs]

    if not documents:
        return "No documents found in the uploaded files."

    binary_embeddings = generate_document_embeddings(documents, embedding_model)
    if not binary_embeddings:
        return "Could not generate embeddings for the documents."

    dim = len(binary_embeddings[0]) * 8

    create_collection_if_not_exists(milvus_client, COLLECTION_NAME, dim)

    data_to_insert = [
        {"context": context, "binary_vector": binary_embedding}
        for context, binary_embedding in zip(documents, binary_embeddings)
    ]
    insert_data(milvus_client, COLLECTION_NAME, data_to_insert)

    return f"Successfully indexed {len(documents)} documents."


def chat_interface(message, history):
    """Chat interface for the RAG pipeline."""
    query_embedding = generate_query_embeddings(message, embedding_model)
    if not query_embedding:
        return "Sorry, I could not process your query."

    contexts = search(milvus_client, COLLECTION_NAME, query_embedding)
    if not contexts:
        return "I couldn't find any relevant information in the documents."

    answer = answer_question(message, contexts)
    return answer


with gr.Blocks() as demo:
    gr.Markdown("## RAG with Binary Quantization")

    with gr.Tab("Upload & Index"):
        file_input = gr.File(file_count="multiple", label="Upload Documents")
        index_button = gr.Button("Update Index")
        index_status = gr.Textbox(label="Indexing Status")

    with gr.Tab("Chat"):
        gr.ChatInterface(chat_interface)

        index_button.click(
            fn=index_documents,
            inputs=[file_input],
            outputs=[index_status],
        )

if __name__ == "__main__":
    # Ensure the documents directory exists from the start
    os.makedirs(DOCS_DIR, exist_ok=True)
    # Reset collection state if there are no documents at startup
    reset_collection_if_no_docs()
    demo.launch()
