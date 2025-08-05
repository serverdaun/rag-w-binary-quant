import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import numpy as np
from pymilvus import MilvusClient, DataType
import logging
from langchain_core.messages import HumanMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


load_dotenv()

DOCS_DIR = "documents"
MODEL_NAME = "gpt-4.1"
TEMPERATURE = 0.2
COLLECTION_NAME = "fast_rag"


def batch_iterate(items, batch_size):
    """Iterate over items in batches."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


llm = init_chat_model(MODEL_NAME, model_provider="openai", temperature=TEMPERATURE)

## Generate binary embeddings
def generate_binary_embeddings():
    """Generate binary embeddings from documents."""
    try:
        # Define loader
        loader = SimpleDirectoryReader(
            input_dir=DOCS_DIR,
            required_exts=[".pdf"],
            recursive=True,
        )

        docs = loader.load_data()
        documents = [doc.text for doc in docs]
        
        if not documents:
            logger.error("No documents found in the documents directory.")
            return [], []

        # Generate embeddings
        embedding_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-large-en-v1.5",
            trust_remote_code=True,
            cache_folder=".hf_cache",
        )

        binary_embeddings = []

        for context in batch_iterate(documents, batch_size=512):
            # generate float32 embeddings
            batch_embeddings = embedding_model.get_text_embedding_batch(context)

            # convert float32 to binary vectors
            embeds_array = np.array(batch_embeddings)
            binary_embeds = np.where(embeds_array > 0, 1, 0).astype(np.uint8)

            # convert to bytes array
            packed_embeds = np.packbits(binary_embeds, axis=1)
            byte_embeds = [vec.tobytes() for vec in packed_embeds]

            binary_embeddings.extend(byte_embeds)
        
        logger.info(f"Generated {len(binary_embeddings)} binary embeddings")
        return documents, binary_embeddings
 
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return [], []


documents, binary_embeddings = generate_binary_embeddings()

## Vector indexing
client = MilvusClient("milvus_binary_quantized.db")

# Initialize client and schema
def create_collection(documents, embeddings):
    try:
        if client.has_collection(COLLECTION_NAME):
            logger.info(f"Collection {COLLECTION_NAME} already exists, dropping it...")
            client.drop_collection(COLLECTION_NAME)

        # Initialize client
        schema = client.create_schema(
            auto_id=True,
            enable_dynamic_fields=True,
        )
    except Exception as e:
        logger.error(f"Error creating collection: {e}")
        return None

    # Add primary key field
    schema.add_field(
        field_name="id",
        datatype=DataType.INT64,
        is_primary=True,
        auto_id=True,
    )
    
    # Add fields to schema
    schema.add_field(
        field_name="context",
        datatype=DataType.VARCHAR,
        max_length=65535,  # max length for VARCHAR
    )
    schema.add_field(
        field_name="binary_vector",
        datatype=DataType.BINARY_VECTOR,
        dim=1024,  # dimension for binary vector
    )

    # Create index params for binary vector
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="binary_vector",
        index_name="binary_vector_index",
        index_type="BIN_FLAT", # Exact search for binary vectors
        metric_type="HAMMING", # Hamming distance for binary vectors
    )

    # Create collection with schema and index
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )

    # Insert data into collection
    client.insert(
        collection_name=COLLECTION_NAME,
        data=[
            {
                "context": context,
                "binary_vector": binary_embedding
            }
            for context, binary_embedding in zip(documents, embeddings)
        ]
    )

create_collection(documents, binary_embeddings)


def get_query_embeddings(query: str) -> bytes:
    """Get query embeddings."""
    try:
        embedding_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-large-en-v1.5",
            trust_remote_code=True,
            cache_folder=".hf_cache",
        )
    except Exception as e:
        logger.error(f"Error getting query embeddings: {e}")
        return None

    # Generate float32 embeddings
    query_embedding = embedding_model.get_text_embedding(query)

    # Convert float32 to binary vector
    binary_vector = np.where(np.array(query_embedding) > 0, 1, 0).astype(np.uint8)

    # Convert to bytes array
    packed_vector = np.packbits(binary_vector, axis=0)

    return packed_vector.tobytes()


def search_documents(query: str, limit: int = 5):
    """Search documents using binary embeddings."""
    try:
        binary_query = get_query_embeddings(query)
        if binary_query is None:
            logger.error("Failed to generate query embeddings")
            return []

        search_results = client.search(
            collection_name=COLLECTION_NAME,
            data=[binary_query],
            anns_field="binary_vector",
            search_params={
                "metric_type": "HAMMING",
            },
            output_fields=["context"],
            limit=limit,
        )

        # logger.info(f"Search results: {search_results}")

        if not search_results:
            logger.error("No search results found")
            return []
        
        contexts = [res.entity.context for res in search_results[0]]

        return contexts

    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        return []


# Test the search functionality
query = "authors of the document"
contexts = search_documents(query, limit=5)

prompt = f"""
# Role and objective
You are a helpful assistant that can answer questions about the following context.

# Intstructions
Given the context information, answer the user's query.
If the context information is not relevant to the user's query, say "I don't know".

# Context
{contexts}

# User's query
{query}

# Answer
"""

human_message = HumanMessage(content=prompt)
print(f"Human message: {human_message}")

response = llm.invoke(input=[human_message])

print(f"Response from the model: {response.content}")
