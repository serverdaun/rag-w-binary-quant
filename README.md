# RAG with Binary Quantization

A high-performance Retrieval-Augmented Generation (RAG) system that uses binary quantization for efficient vector storage and similarity search. This project implements a document Q&A system with optimized memory usage and fast retrieval capabilities.

## 🚀 Features

- **Binary Quantization**: Converts high-dimensional embeddings to binary vectors for memory efficiency
- **Milvus Vector Database**: Uses Milvus for scalable vector storage and similarity search
- **Gradio Web Interface**: User-friendly web UI for document upload and chat
- **BGE Embeddings**: Leverages BAAI/bge-large-en-v1.5 for high-quality text embeddings
- **OpenAI Integration**: Uses GPT-4.1 for intelligent question answering
- **Batch Processing**: Efficient document processing with configurable batch sizes

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Documents     │───▶│  BGE Embeddings  │───▶│ Binary Vectors  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  Query Embedding │───▶│  Milvus Search  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Retrieved Docs │◀───│  Context Fusion  │◀───│  LLM Answer     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd rag-w-binary-quant
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Set up environment variables**:
   Create a `.env` file with your OpenAI API key:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## 🚀 Usage

### Starting the Application

Run the Gradio web interface:
```bash
uv run app.py
```

The application will be available at `http://localhost:7860`

### Using the Interface

1. **Upload Documents**: 
   - Go to the "Upload & Index" tab
   - Upload your documents (supports multiple file formats)
   - Click "Update Index" to process and index the documents

2. **Chat with Documents**:
   - Switch to the "Chat" tab
   - Ask questions about your uploaded documents
   - Get intelligent answers based on the document content

## 🔧 Configuration

Key configuration parameters in `src/config.py`:

- `EMBEDDING_MODEL_NAME`: BAAI/bge-large-en-v1.5
- `COLLECTION_NAME`: "fast_rag"
- `MILVUS_DB_PATH`: "milvus_binary_quantized.db"
- `MODEL_NAME`: "gpt-4.1"
- `TEMPERATURE`: 0.2

## 📊 Performance Benefits

- **Memory Efficiency**: Binary vectors use 8x less memory than float32 embeddings
- **Fast Search**: Hamming distance computation is highly optimized
- **Scalable**: Milvus provides enterprise-grade vector database capabilities
- **Accurate**: BGE embeddings provide high-quality semantic representations

## 🏛️ Project Structure

```
rag-w-binary-quant/
├── app.py                 # Gradio web interface
├── main.py               # Main application entry point
├── src/
│   ├── config.py         # Configuration settings
│   ├── data_loader.py    # Document loading utilities
│   ├── embedding_generator.py  # Binary embedding generation
│   ├── vector_store.py   # Milvus vector database operations
│   └── rag_pipeline.py   # RAG question answering pipeline
├── documents/            # Uploaded document storage
└── README.md
```

## 🔍 Technical Details

### Binary Quantization Process

1. **Float32 Embeddings**: Generate embeddings using BGE model
2. **Binary Conversion**: Convert to binary using threshold (positive values → 1, negative → 0)
3. **Packing**: Pack binary vectors into bytes for efficient storage
4. **Hamming Distance**: Use Hamming distance for similarity search

### Vector Search

- **Index Type**: BIN_FLAT (exact search for binary vectors)
- **Metric**: Hamming distance
- **Retrieval**: Top-k most similar documents

## 🙏 Acknowledgments

- [BAAI](https://github.com/FlagOpen/FlagEmbedding) for the BGE embedding model
- [Milvus](https://milvus.io/) for the vector database
- [Gradio](https://gradio.app/) for the web interface
- [OpenAI](https://openai.com/) for the language model
