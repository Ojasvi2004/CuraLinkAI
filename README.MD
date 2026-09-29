rag-service/
├── .github/
│   └── workflows/
│       ├── ci.yml                     # Linting, typing, pytest
│       └── cd.yml                     # Container build & deployment
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml             # Local infra (Redis, Qdrant/Pinecone mock)
│   └── .dockerignore
├── configs/
│   ├── config.yaml                    # Model names, top_k, thresholds, chunk sizes
│   └── logging.yaml                   # Structured JSON logging config
├── src/
│   ├── __init__.py
│   ├── main.py                        # FastAPI entrypoint, middleware, health probes
│   │
│   ├── api/                           # Delivery / Transport Layer
│   │   ├── __init__.py
│   │   ├── deps.py                    # DI: DB sessions, client singletons, rate limiters
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── router.py              # APIRouter aggregation
│   │   │   └── endpoints/
│   │   │       ├── ingest.py          # File upload & background parsing trigger
│   │   │       ├── query.py           # RAG retrieval & SSE streaming chat
│   │   │       └── health.py          # /healthz and /readyz probes
│   │   └── middlewares/
│   │       ├── correlation_id.py      # Tracing X-Request-ID across spans
│   │       └── error_handler.py       # Global exception to RFC 7807 response
│   │
│   ├── core/                          # Cross-Cutting Infrastructure
│   │   ├── __init__.py
│   │   ├── config.py                  # Pydantic BaseSettings (reads env vars)
│   │   ├── logger.py                  # Structlog / Loguru JSON logger setup
│   │   └── exceptions.py              # Custom domain exceptions
│   │
│   ├── schemas/                       # Pydantic DTOs (Request / Response validation)
│   │   ├── __init__.py
│   │   ├── ingest.py                  # DocumentUploadRequest, IngestResponse
│   │   ├── query.py                   # QueryRequest, SourceCitation, ChatResponse
│   │   └── common.py                  # Pagination, standard envelope wrappers
│   │
│   ├── services/                      # Application Business Logic
│   │   ├── __init__.py
│   │   ├── ingest_pipeline.py         # Orchestrates: Extract -> Chunk -> Embed -> Store
│   │   └── rag_pipeline.py            # Orchestrates: Rewrite -> Retrieve -> Rerank -> Synthesize
│   │
│   ├── components/                    # Modular RAG Building Blocks
│   │   ├── __init__.py
│   │   ├── parsers/                   # Document extraction
│   │   │   ├── base.py                # Abstract Base Class (ABC) for parsers
│   │   │   ├── pdf_parser.py          # PyMuPDF / pdfplumber parser
│   │   │   └── ocr_parser.py          # PyTesseract / Tesseract subprocess worker
│   │   │
│   │   ├── chunking/                  # Segmentation strategies
│   │   │   ├── base.py
│   │   │   ├── recursive.py           # Recursive character-level splitter
│   │   │   └── semantic.py            # Sentence-boundary / semantic splitter
│   │   │
│   │   ├── embeddings/                # Vector representations
│   │   │   ├── base.py
│   │   │   ├── local_bge.py           # Local SentenceTransformers (if self-hosted)
│   │   │   └── remote_client.py       # HF Inference API / OpenAI / Cohere client
│   │   │
│   │   ├── vector_store/              # Vector database abstraction
│   │   │   ├── base.py                # VectorStore ABC (upsert, query, delete)
│   │   │   └── pinecone_store.py      # Pinecone client implementation
│   │   │
│   │   ├── retrievers/                # Search logic
│   │   │   ├── dense.py               # Pure vector cosine similarity
│   │   │   ├── sparse.py              # BM25 / SPLADE lexical index
│   │   │   └── hybrid.py              # Reciprocal Rank Fusion (RRF) implementation
│   │   │
│   │   ├── rerankers/                 # Cross-encoder precision refinement
│   │   │   ├── base.py
│   │   │   └── cross_encoder.py       # Cohere Rerank / BGE-Reranker client
│   │   │
│   │   └── generators/                # LLM output synthesis
│   │       ├── base.py
│   │       ├── prompts.py             # Jinja2 / strict formatted prompt templates
│   │       └── llm_client.py          # Streaming LLM adapter (OpenAI / Anthropic / vLLM)
│   │
│   └── utils/
│       ├── __init__.py
│       ├── telemetry.py               # OpenTelemetry / Langfuse instrumentation
│       └── text_cleaning.py           # Unicode normalization & regex sanitization
│
├── tests/
│   ├── conftest.py                    # Fixtures: mock vector DB, sample chunks
│   ├── unit/
│   │   ├── test_chunking.py
│   │   ├── test_parsers.py
│   │   └── test_embeddings.py
│   ├── integration/
│   │   ├── test_pinecone_ops.py
│   │   └── test_api_routes.py
│   └── evaluation/
│       └── test_ragas_metrics.py      # Faithfulness, answer relevance, context recall
│
├── .env.example
├── pyproject.toml                     # Poetry / uv dependency declarations
└── README.md