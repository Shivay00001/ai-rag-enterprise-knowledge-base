# AI RAG Enterprise Knowledge Base

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1-orange.svg)](https://langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-grade **Retrieval-Augmented Generation (RAG)** system for enterprise knowledge management. Ingest documents, build semantic indexes, and query your knowledge base using natural language.

## 🚀 Features

- **Multi-format Document Ingestion**: PDF, DOCX, TXT, Markdown, HTML
- **Semantic Search**: ChromaDB vector store with sentence-transformers embeddings
- **LLM Integration**: OpenAI GPT-4, Anthropic Claude, or local models via Ollama
- **REST API**: FastAPI with OpenAPI documentation
- **Authentication**: JWT-based auth with role-based access control
- **Async Processing**: Background document processing with Celery
- **Observability**: Structured logging, Prometheus metrics, health checks

## 📁 Project Structure

```
ai-rag-enterprise-knowledge-base/
├── src/
│   ├── api/                 # FastAPI routes
│   │   ├── __init__.py
│   │   ├── documents.py     # Document upload/management
│   │   ├── query.py         # RAG query endpoints
│   │   └── auth.py          # Authentication
│   ├── core/                # Core RAG logic
│   │   ├── __init__.py
│   │   ├── embeddings.py    # Embedding models
│   │   ├── vectorstore.py   # ChromaDB integration
│   │   ├── retriever.py     # Document retrieval
│   │   ├── llm.py           # LLM abstraction
│   │   └── chains.py        # LangChain RAG chains
│   ├── ingestion/           # Document processing
│   │   ├── __init__.py
│   │   ├── loaders.py       # File loaders
│   │   ├── chunkers.py      # Text splitting
│   │   └── pipeline.py      # Ingestion pipeline
│   ├── models/              # Pydantic models
│   │   ├── __init__.py
│   │   ├── documents.py
│   │   └── queries.py
│   ├── config.py            # Configuration
│   └── main.py              # Application entrypoint
├── tests/
│   ├── conftest.py
│   ├── test_api/
│   ├── test_core/
│   └── test_ingestion/
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── requirements.txt
└── README.md
```

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/Shivay00001/ai-rag-enterprise-knowledge-base.git
cd ai-rag-enterprise-knowledge-base

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the application
uvicorn src.main:app --reload
```

## 🐳 Docker

```bash
docker-compose up -d
```

## 📖 API Usage

### Upload Document

```bash
curl -X POST "http://localhost:8000/api/v1/documents" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@knowledge_base.pdf"
```

### Query Knowledge Base

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is our refund policy?", "top_k": 5}'
```

## ⚙️ Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | - |
| `EMBEDDING_MODEL` | Embedding model name | `all-MiniLM-L6-v2` |
| `LLM_MODEL` | LLM model name | `gpt-4-turbo-preview` |
| `CHROMA_PERSIST_DIR` | ChromaDB storage path | `./data/chroma` |
| `CHUNK_SIZE` | Document chunk size | `1000` |
| `CHUNK_OVERLAP` | Chunk overlap | `200` |

## 🧪 Testing

```bash
pytest tests/ -v --cov=src
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
