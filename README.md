# 🌀 AI Terminal - OpenAI SDK Chatbot

A context-aware AI chatbot that uses Retrieval Augmented Generation (RAG) to answer questions based on your custom documents. Built with OpenAI's API, FAISS for semantic search, and Python.

## ✨ Features

- **Semantic Search**: Uses OpenAI embeddings with FAISS to find relevant documents
- **Context-Aware Answers**: Augments user queries with retrieved document context
- **Interactive CLI**: Real-time conversation with command support
- **Dynamic Memory Building**: Index your documents for intelligent question answering
- **Flexible Configuration**: Environment-based model selection (GPT-4o, GPT-4o-mini, etc.)

## 📁 Project Structure

```
open_ai_sample/
├── ai_brain.py          # Core AI logic: query memory and generate responses
├── memory_index.py      # FAISS indexing: embeddings and semantic search
├── main.py              # Interactive terminal interface
├── config.py            # Configuration and environment variables
├── requirements.txt     # Python dependencies
└── RAG/                 # Documents to index for context
    ├── *.sql            # SQL files (indexed)
    ├── *.md             # Markdown files (indexed)
    └── *.csv            # Data files (indexed)
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key (with embeddings and chat models access)

### Installation

1. **Clone the repository**
   ```bash
   cd c:\Repos\open_ai_sample
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set environment variables**
   ```bash
   # Windows (PowerShell)
   $env:OPENAI_API_KEY = "your-api-key-here"
   
   # Windows (Command Prompt)
   set OPENAI_API_KEY=your-api-key-here
   
   # Linux/macOS
   export OPENAI_API_KEY="your-api-key-here"
   ```

   **Optional**: Customize models
   ```bash
   $env:OPENAI_CHAT_MODEL = "gpt-4o"           # default: gpt-4o
   $env:OPENAI_EMBED_MODEL = "text-embedding-3-small"  # default
   ```

## 💬 Usage

Start the interactive AI terminal:

```bash
python main.py
```

Then chat with the AI:
```
You: What are the sales trends?
AI: [Response based on your RAG documents]

You: /reload
🔁 Memory index rebuilt.

You: /exit
👋 Goodbye, see you next time!
```

### Commands

| Command | Description |
|---------|-------------|
| `/reload` | Rebuild the FAISS memory index from RAG documents |
| `/exit` | Exit the terminal |

## 🔧 Configuration

Edit `config.py` to adjust settings or set environment variables:

```python
OPENAI_API_KEY              # Your OpenAI API key (required)
OPENAI_CHAT_MODEL          # Chat model (default: gpt-4o)
OPENAI_EMBED_MODEL         # Embedding model (default: text-embedding-3-small)
```

## 📚 How It Works

1. **Indexing** (`memory_index.py`):
   - Scans the `RAG/` directory for `.md`, `.txt`, `.json`, `.sql`, and `.csv` files
   - Chunks documents into overlapping segments (1800 chars with 200 char overlap)
   - Generates embeddings using OpenAI's embedding API
   - Stores vectors in a FAISS index for fast semantic search

2. **Query** (`ai_brain.py`):
   - Embeds your question
   - Retrieves top-k relevant document chunks from FAISS
   - Sends question + context to GPT model
   - Returns AI-generated answer

3. **Interaction** (`main.py`):
   - Provides a REPL for continuous conversation
   - Supports special commands (`/reload`, `/exit`)

## 📦 Dependencies

- **openai** (>=1.40.0) - OpenAI API client
- **faiss-cpu** (>=1.8.0) - Similarity search and clustering
- **numpy** (>=1.26) - Numerical computing

For GPU support, use `faiss-gpu` instead of `faiss-cpu`.

## 🎯 Example: Adding Documents

Place your documents in the `RAG/` folder:

```
RAG/
├── File_A.SQL     # Database schema, queries
├── File_B.SQL     # More SQL docs
├── File_C.SQL     # Additional data
├── Hello.md       # Markdown notes
└── Sales.csv      # CSV data
```

Then restart and type `/reload`:
```
You: /reload
🔁 Memory index rebuilt.
```

## ⚙️ Advanced Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_FILE_CHARS` | 200,000 | Max characters per file |
| `CHUNK_CHARS` | 1,800 | Chunk size for indexing |
| `CHUNK_OVERLAP` | 200 | Overlap between chunks |
| `EMBED_BATCH` | 64 | Batch size for embeddings |
| `top_k` | 5 | Retrieval results count |

Edit `memory_index.py` constants to adjust these values.

## 🛠️ Troubleshooting

**❌ "RAG directory not found"**
- Ensure the `RAG/` folder exists in the project root

**❌ "No embeddings returned"**
- Check your OpenAI API key is valid
- Verify you have embeddings/chat quota
- Ensure documents exist and meet minimum size (50 chars)

**❌ "No relevant memory found"**
- Your documents may not contain relevant information
- Try adding more documents to the `RAG/` folder
- Use `/reload` to rebuild the index

## 📝 License

MIT License - feel free to use and modify

## 🤝 Contributing

Suggestions and improvements welcome!
