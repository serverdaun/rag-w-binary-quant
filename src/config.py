DOCS_DIR = "documents"
MODEL_NAME = "gpt-4.1"
MODEL_PROVIDER = "openai"
TEMPERATURE = 0.2
COLLECTION_NAME = "fast_rag"
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
MILVUS_DB_PATH = "milvus_binary_quantized.db"

PROMPT = """ # Role and objective
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