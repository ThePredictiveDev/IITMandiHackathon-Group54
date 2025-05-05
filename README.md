# MATLAB / Simulink Real‑Time Troubleshooter

> **🏆 3rd Place Winner - IIT Mandi Deep Learning Hackathon 2025**  
> *A Multi-Agent Hierarchical Corrective RAG Architecture for MATLAB & Simulink Documentation*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Redis](https://img.shields.io/badge/redis-required-red.svg)](https://redis.io/)
[![FAISS](https://img.shields.io/badge/FAISS-CPU/GPU-orange.svg)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📑 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Technical Details](#-technical-details)
- [Directory Structure](#-directory-structure)
- [Prerequisites & Dependencies](#-prerequisites--dependencies)
- [Installation & Setup](#-installation--setup)
- [Running the Application](#-running-the-application)
- [Configuration Parameters](#-configuration-parameters)
- [Troubleshooting](#-troubleshooting)
- [Performance Optimization](#-performance-optimization)
- [Contribution Guidelines](#-contribution-guidelines)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

## 🌟 Overview

This repository contains a complete multi-agent Retrieval-Augmented Generation (RAG) stack designed to ingest MathWorks documentation and power an interactive troubleshooting assistant for MATLAB and Simulink users. The system provides full chain-of-thought transparency at every step of the reasoning process.

What makes this project unique:
- **Multi-Agent Architecture**: Four distinct LLM calls orchestrated in a hierarchical pipeline
- **Explicit Chain-of-Thought**: Full reasoning transparency at each processing stage
- **Novel Chunk Re-Ranking**: LLM-driven semantic evaluation with explicit reasoning
- **Verification System**: Adaptive verification thresholds with retry logic
- **Dual-Tier Memory**: Short-term context and long-term Redis-backed memory

## 🏗 Architecture

![Architecture Diagram](https://i.imgur.com/TBAeqSl.png)

### 4-Stage LLM Orchestration Pipeline

1. **Planner Agent** (Llama-3-8B-8192)
   - Analyzes query to determine optimal k value for retrieval
   - Extracts salient domain-specific keywords
   - Filters off-topic questions outside MATLAB/Simulink domain
   - Produces planning chain-of-thought reasoning

2. **Chunk Re-Ranking Agent** (Llama-3-8B-8192)
   - Re-scores vector-retrieved chunks for semantic relevance
   - Provides explicit reasoning for each ranking decision
   - Filters out irrelevant chunks despite vector similarity

3. **Writer Agent** (DeepSeek-R1-Distill-Llama-70B)
   - Produces structured responses with three distinct sections:
     - `THOUGHT`: Internal reasoning summary
     - `ACTION`: Concrete troubleshooting steps
     - `EVIDENCE`: Citations with documentation links

4. **Verifier Agent** (Llama-3.1-8B-Instant)
   - Validates writer output against original query
   - Ensures factual grounding in retrieved documentation
   - Implements adaptive verification thresholds with retry logic

### Memory & Caching Architecture

- **Short-Term Memory**: In-process deque of recent conversation turns
- **Long-Term Memory**: Redis-backed storage with TTL-based eviction
- **Response Cache**: Fast-path Redis cache (15-minute TTL) for frequent queries

### Vector Search System

- **Embeddings**: E5-small-v2 transformer model
- **Index**: FAISS-HNSW (Hierarchical Navigable Small World)
- **Customizable Parameters**: M (graph connectivity), efConstruction, efSearch

## 🔬 Technical Details

### Data Processing Pipeline

1. **Document Cleaning** (`prep_matlab_docs.py`)
   - Removes boilerplate content and navigation elements
   - Deduplicates sentences and normalizes whitespace
   - Extracts MathWorks-specific technical tags and error codes
   - Applies language detection to filter non-English content
   - Outputs JSON with content hash for deduplication

2. **Document Chunking** (`chunk_docs.py`)
   - Tokenizes content using model-specific tokenizers
   - Implements window-chunking with 128-token segments
   - Applies 32-token stride for overlapping contexts
   - Manages token counts with either HuggingFace or TikToken
   - Preserves metadata including URLs and technical tags

3. **Index Building** (`build_index.py`)
   - Embeds chunks with E5-small-v2 model
   - Constructs FAISS-HNSW index with configurable parameters
   - Implements normalization for cosine similarity
   - Supports both CPU and GPU acceleration
   - Saves embedding cache for rapid rebuilding

### LLM Chain Orchestration

1. **Query Planning** (`planner_agent.py`)
   - Implements few-shot examples for consistent output schema
   - Balances k between 4 (simple queries) and 8+ (complex queries)
   - Provides public and private reasoning chains
   - Handles off-topic detection with explicit rejection
   - Re-ranks chunks with explicit reasoning for each decision

2. **Response Generation** (`writer_agent.py`)
   - Structured output with THOUGHT/ACTION/EVIDENCE sections
   - Token-based streaming for responsive UI
   - Explicit section markers for reliable parsing
   - Numbered evidence with source URL citations
   - Temperature controls for creativity vs. precision balance

3. **Response Verification** (`verifier_agent.py`)
   - Iterative verification with increasing leniency
   - Timeout-based retry logic (minimum 5 iterations or 60 seconds)
   - JSON-structured verification decision and reasoning
   - Graceful fallback to best attempt if no verification succeeds

### Memory & Caching System

1. **Short-Term Memory** (`memory.py`)
   - Configurable maximum turn count (default: 10)
   - In-process deque with O(1) operations
   - Role-based storage (user/assistant)
   - Timestamp tracking for recency

2. **Long-Term Memory** (`memory.py`)
   - Redis sorted set with score-based timestamp
   - TTL-based automatic eviction (default: 400 minutes)
   - Size-capped storage (default: 1000 entries maximum)
   - Content-hashed for deduplication

3. **Response Cache** (`cache.py`)
   - Query-hashed lookup system
   - 15-minute TTL for freshness
   - Graceful fallback to in-memory cache on Redis failure
   - Minimal serialization overhead with direct JSON storage

### Frontend & UI

1. **Gradio Interface** (`frontend.py`)
   - Responsive chat-like interface with custom CSS
   - Collapsible panels for memory, chain-of-thought, and detailed logs
   - JSON logging for complete transparency
   - Memory and cache management controls
   - Section parsing and display formatting

2. **CLI Interface** (`chatbot_dep.py`)
   - Terminal-based interactive chat
   - Complete pipeline visibility with stage-by-stage output
   - Raw JSON diagnostic output
   - Memory query intent detection

3. **Batch Evaluation** (`batch_chatbot_demo.py`)
   - Automated testing of predefined queries
   - Structured output formatting
   - Performance logging
   - Results aggregation in text format

## 📂 Directory Structure

```
/ (root)
├── agents/                        # LLM agent implementations
│   ├── planner_agent.py           # Query planning, chunk scoring
│   ├── writer_agent.py            # Response generation
│   └── verifier_agent.py          # Solution verification
│
├── data preprocessing/            # Preprocessing pipeline
│   ├── prep_matlab_docs.py        # Document cleaning, tag extraction
│   └── chunk_docs.py              # Tokenization, chunking, strides
│
├── index_tools_build_and_retrieve/ # Embedding and retrieval
│   ├── build_index.py             # FAISS index construction
│   └── retrieval.py               # Vector search + semantic reranking
│
├── stores_mem_and_cache/          # Storage subsystems
│   ├── memory.py                  # STM/LTM implementation
│   └── cache.py                   # Response caching
│
├── data (json+index+raw csv)/     # Data storage (gitignored)
│   ├── raw_data.csv               # Source MathWorks documentation CSV
│   ├── clean_docs.jsonl           # Clean, deduplicated documents
│   ├── docs_chunks.jsonl          # Chunked documents with metadata
│   ├── faiss.index                # FAISS vector index
│   ├── metadata.jsonl             # Chunk metadata for retrieval
│   └── embeddings.npy             # Cached embeddings for fast rebuilding
│
├── results/                       # Evaluation outputs
│   └── batch_results.txt          # Results from batch testing
│
├── sample questions/              # Test data
│   └── simulink_questions.txt     # Curated test questions
│
├── test scripts/                  # Diagnostics and validation
│   ├── test_memory.py             # Memory system diagnostics
│   └── test_verifier.py           # Verification system tests
│
├── frontend.py                    # Gradio UI implementation
├── chatbot_dep.py                 # Core orchestration logic
├── batch_chatbot_demo.py          # Batch evaluation runner
└── requirements.txt               # Dependencies
```

## 🧩 Prerequisites & Dependencies

### System Requirements

- Python 3.8+ (3.10+ recommended)
- 8GB+ RAM (16GB+ recommended for larger indices)
- CUDA-compatible GPU (optional, for faster embeddings)
- Redis server 6.0+ (for memory and caching)

### Core Dependencies

```
# LLM & HTTP clients
groq>=0.8.0          # Groq API client for LLM access
httpx>=0.23.0        # Async HTTP client

# Caching & memory
redis>=4.5.0         # Redis client for distributed memory/cache
langdetect>=1.0.9    # Language detection for preprocessing

# Retrieval & embeddings
faiss-cpu>=1.7.4     # Vector search (or faiss-gpu)
sentence-transformers>=2.2.2  # Embedding models

# Deep learning backends
torch>=2.0.0         # PyTorch for embeddings
tensorflow>=2.19.0   # TensorFlow (optional for some models)

# Frontend
gradio>=5.29.0       # UI framework

# Utilities
numpy>=1.24.0        # Numerical operations
```

### API Keys

- **Groq API Key**: Required for LLM access
  - Set as environment variable `GROQ_API_KEY`
  - Alternatively, hardcoded in agent files for demo purposes

## 🔧 Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/ThePredictiveDev/IITMandiHackathon-Group54.git
cd IITMandiHackathon-Group54
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Or using conda
conda create -n matlab-troubleshooter python=3.10
conda activate matlab-troubleshooter
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

# For GPU acceleration (optional)
pip uninstall -y faiss-cpu
pip install faiss-gpu
```

### 4. Set Up Redis

```bash
# Install Redis
# On Ubuntu/Debian
sudo apt-get install redis-server

# On macOS
brew install redis

# On Windows
# Download from https://github.com/microsoftarchive/redis/releases

# Start Redis server
redis-server
```

### 5. Set Environment Variables

```bash
# Linux/Mac
export GROQ_API_KEY="your_groq_api_key_here"
export REDIS_HOST="localhost"
export REDIS_PORT=6379
export REDIS_DB_MEMORY=1

# Windows
set GROQ_API_KEY=your_groq_api_key_here
set REDIS_HOST=localhost
set REDIS_PORT=6379
set REDIS_DB_MEMORY=1
```

### 6. Prepare Data & Build Index

```bash
# 1. Clean raw MathWorks documentation
python "data preprocessing/prep_matlab_docs.py" \
  --csv "data (json+index+raw csv)/raw_data.csv" \
  --out "data (json+index+raw csv)/clean_docs.jsonl" \
  --min-tokens 50

# 2. Chunk documents
python "data preprocessing/chunk_docs.py" \
  --input "data (json+index+raw csv)/clean_docs.jsonl" \
  --output "data (json+index+raw csv)/docs_chunks.jsonl" \
  --chunk-size 128 \
  --stride 32

# 3. Build FAISS index
python "index_tools_build_and_retrieve/build_index.py" \
  --chunks "data (json+index+raw csv)/docs_chunks.jsonl" \
  --index "data (json+index+raw csv)/faiss.index" \
  --meta "data (json+index+raw csv)/metadata.jsonl" \
  --cache "data (json+index+raw csv)/embeddings.npy" \
  --M 32 \
  --ef-constr 64 \
  --ef-search 128
```

## 🚀 Running the Application

### Interactive CLI Mode

```bash
python chatbot_dep.py
```

The CLI interface allows interactive querying with detailed visibility into each pipeline stage:
- Planner reasoning and query analysis
- Chunk selection with scoring rationale
- Writer drafting with THOUGHT/ACTION/EVIDENCE structure
- Verification process and decisions

Enter questions at the prompt and type "exit" or "quit" to terminate.

### Gradio Web Interface

```bash
python frontend.py
```

The web interface offers:
- Chat-like interaction with the troubleshooter
- Collapsible panels for memory inspection
- Full chain-of-thought visibility
- Detailed JSON logs for pipeline analysis
- Memory and cache management controls

By default, the interface launches on http://localhost:7860

To share publicly (e.g., for demos):

```bash
python frontend.py --share=True
```

### Batch Evaluation

```bash
python batch_chatbot_demo.py
```

Runs the system through 15 diverse Simulink/MATLAB queries and outputs results to `batch_results.txt`:
- Standardized output format for consistent evaluation
- Raw reasoning chains for comparison
- ACTION steps for troubleshooting validation
- EVIDENCE with citations to verify accuracy

## ⚙️ Configuration Parameters

### Memory System

| Parameter | Default | Description |
|-----------|---------|-------------|
| `STM_MAX_TURNS` | 10 | Maximum number of turns in short-term memory |
| `LTM_TTL_SECONDS` | 24000 (400 min) | Time-to-live for long-term memory entries |
| `LTM_MAX_ENTRIES` | 1000 | Maximum number of entries in long-term memory |

### Index Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `M` | 32 | Graph connectivity in HNSW (higher = more accurate, slower) |
| `efConstruction` | 64 | Index build quality parameter (higher = better index, slower build) |
| `efSearch` | 128 | Search quality parameter (higher = more accurate, slower search) |

### LLM Configuration

| Agent | Model | Temperature | Max Tokens | Purpose |
|-------|-------|-------------|------------|---------|
| Planner | llama3-8b-8192 | 0.0 | 512 | Query analysis, chunk scoring |
| Writer | deepseek-r1-distill-llama-70b | 0.3 | 2048 | Response generation |
| Verifier | llama-3.1-8b-instant | 0.0 | 512 | Solution verification |

### Chunking Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 128 | Token length of each chunk |
| `stride` | 32 | Overlap between consecutive chunks |

## 🔍 Troubleshooting

### Common Issues

#### Redis Connection Failures

```
[cache] Redis error (Error 111 connecting to localhost:6379. Connection refused.); falling back to local cache
```

**Solution**: 
```bash
# Check if Redis is running
redis-cli ping

# If not, start Redis
redis-server --daemonize yes
```

#### FAISS Index Loading Errors

```
OSError: Error loading FAISS index: <reason>
```

**Solutions**:
- Ensure the index path is correct
- Rebuild the index with compatible parameters
- Check for GPU/CPU mismatch in FAISS installation

#### LLM API Rate Limiting

```
Error from Groq API: rate_limit_exceeded
```

**Solutions**:
- Implement exponential backoff retry logic
- Reduce batch sizes for evaluation
- Use a different API key with higher limits

#### Vector Dimension Mismatch

```
RuntimeError: FAISS index has dimension X but embeddings have dimension Y
```

**Solution**: Rebuild the index with the same model used for query embeddings

### Diagnostic Tools

```bash
# Test memory system
python "test scripts/test_memory.py"

# Test verification logic
python "test scripts/test_verifier.py"

# Test Redis connection
redis-cli ping
```

## 🚄 Performance Optimization

### CPU vs. GPU Acceleration

The system supports both CPU and GPU modes:

```python
# In build_index.py and retrieval.py:
EMBED_DEVICE = "cuda" if faiss.get_num_gpus() > 0 else "cpu"
```

For GPU acceleration:
```bash
pip uninstall -y faiss-cpu
pip install faiss-gpu
```

### Memory Usage Considerations

- **Index Size**: Scales linearly with document count and embedding dimension
- **Redis Memory**: Configure `maxmemory` and eviction policies in `redis.conf`
- **Embedding Cache**: Can be disabled for low-memory environments

### Throughput Optimization

- **Batch Size**: Adjust embedding batch size based on available memory
- **M Parameter**: Lower values trade accuracy for speed in HNSW
- **Caching**: Hot-path response cache significantly improves repeat query performance
- **Index Quantization**: For very large indices, consider INT8 quantization

## 👨‍💻 Contribution Guidelines

### Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Document functions with docstrings
- Use meaningful variable names
- Implement proper error handling

### Adding New Features

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Implement your changes with tests
3. Update documentation as needed
4. Submit a pull request with a clear description

### Testing

- Add unit tests for new functionality
- Ensure compatibility with the existing pipeline
- Validate with real-world MATLAB/Simulink queries

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Competition**: IIT Mandi Deep Learning Hackathon 2023
- **Model Providers**: Groq for API access to powerful LLMs
- **Libraries**: FAISS, Sentence Transformers, Redis, Gradio
- **Documentation**: MathWorks for MATLAB/Simulink documentation corpus

---

*Project ideated and designed by ThePredictiveDev and coded with vibes.*
