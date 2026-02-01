# Enterprise Knowledge Assistant

🚀 **Production-grade intelligent assistant for enterprise document analysis with Agentic RAG and MLOps**

## Overview

A sophisticated multi-agent RAG (Retrieval-Augmented Generation) system designed for legal/compliance teams to query company policies, contracts, and regulations with high accuracy and explainability.

### Key Features

- **🔍 Three-Way Hybrid Retrieval**: Combines BM25, Dense Vectors (BGE-M3), and SPLADE sparse vectors with ColBERT reranking
- **🤖 Multi-Agent Architecture**: LangGraph-based orchestration with specialized agents for planning, extraction, QA, and validation
- **📊 Knowledge Graph Integration**: Neo4j for structured knowledge representation and multi-hop reasoning
- **🔧 Configurable LLM Client**: Works with any OpenAI-compatible API (OpenAI, Azure, Ollama, vLLM, Anthropic, etc.)
- **📈 Full MLOps Pipeline**: DVC, MLflow, Airflow, RAGAS evaluation, automated retraining
- **🔐 Enterprise Security**: JWT auth, RBAC, rate limiting, PII masking
- **📉 Comprehensive Monitoring**: Prometheus, Grafana, distributed tracing

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     API Gateway (FastAPI)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Multi-Agent Orchestrator                     │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐        │   │
│  │  │ Planner │ │Extractor│ │   QA    │ │Validator│        │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│  ┌───────────────────────────▼───────────────────────────────┐  │
│  │              Hybrid Retrieval Pipeline                     │  │
│  │  ┌──────┐  ┌──────────┐  ┌──────┐  ┌─────────┐           │  │
│  │  │ BM25 │  │Dense Vec │  │SPLADE│  │ ColBERT │           │  │
│  │  └──────┘  └──────────┘  └──────┘  │Reranker │           │  │
│  │      └──────────┬───────────┘      └─────────┘           │  │
│  │         RRF Fusion (k=60)                                 │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│  ┌───────────────────────────▼───────────────────────────────┐  │
│  │                    Data Layer                              │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                │  │
│  │  │  Qdrant  │  │  Neo4j   │  │  Redis   │                │  │
│  │  │(Vectors) │  │  (Graph) │  │ (Cache)  │                │  │
│  │  └──────────┘  └──────────┘  └──────────┘                │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- CUDA-compatible GPU (recommended for embeddings)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/enterprise-knowledge-assistant.git
cd enterprise-knowledge-assistant

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -e ".[all]"

# Copy environment template
cp .env.example .env
# Edit .env with your configuration
```

### Start Infrastructure

```bash
# Start Qdrant, Neo4j, Redis
docker-compose up -d

# Verify services are running
docker-compose ps
```

### Run the API

```bash
# Development mode
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn src.api.main:app --workers 4 --host 0.0.0.0 --port 8000
```

### API Documentation

Once running, access:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## LLM Configuration

The system supports any LLM with an OpenAI-compatible API:

### OpenAI
```env
LLM_PROVIDER=openai
LLM_API_BASE=https://api.openai.com/v1
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4-turbo
```

### Azure OpenAI
```env
LLM_PROVIDER=azure
LLM_API_BASE=https://your-resource.openai.azure.com
LLM_API_KEY=your-key
LLM_MODEL=your-deployment
LLM_API_VERSION=2024-02-15-preview
```

### Local Ollama
```env
LLM_PROVIDER=ollama
LLM_API_BASE=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=llama3.1:70b
```

### vLLM / Custom
```env
LLM_PROVIDER=custom
LLM_API_BASE=http://localhost:8000/v1
LLM_API_KEY=not-needed
LLM_MODEL=meta-llama/Meta-Llama-3.1-70B-Instruct
```

## Project Structure

```
enterprise-knowledge-assistant/
├── src/
│   ├── agents/           # Multi-agent system
│   ├── api/              # FastAPI application
│   ├── models/           # LLM & embedding clients
│   ├── retrieval/        # Hybrid retrieval pipeline
│   ├── pipeline/         # Document processing
│   └── monitoring/       # Metrics & tracing
├── infrastructure/
│   ├── docker/           # Docker configurations
│   ├── kubernetes/       # K8s manifests
│   └── terraform/        # IaC for AWS
├── mlops/
│   ├── airflow/          # DAGs for pipelines
│   ├── mlflow/           # Experiment tracking
│   └── dvc/              # Data versioning
├── monitoring/
│   ├── prometheus/       # Metrics collection
│   └── grafana/          # Dashboards
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
└── docs/
```

## Evaluation Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Context Precision | ≥ 0.85 | Retrieved context relevance |
| Context Recall | ≥ 0.90 | Coverage of relevant information |
| Faithfulness | ≥ 0.95 | Answer grounded in context |
| Answer Relevance | ≥ 0.88 | Response quality |
| P95 Latency | < 3s | End-to-end response time |
| Availability | 99.9% | System uptime |

## Development

```bash
# Run tests
pytest tests/ -v

# Run linting
ruff check src/

# Run type checking
mypy src/

# Run all checks
pre-commit run --all-files
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details.
