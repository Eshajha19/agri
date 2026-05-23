# Quick Start: Semantic Chunking

## Installation

```bash
# Install required dependency
pip install tiktoken
```

## Basic Usage

### 1. Enable Semantic Chunking (Default)

```python
from rag.retriever import get_retriever

# Semantic chunking is enabled by default
retriever = get_retriever(use_chunking=True)

# Use the retriever as normal
results = retriever.retrieve("your query here", top_k=3)
```

### 2. Chunk a Document

```python
from rag.chunking import SemanticChunker

# Create chunker with default settings
chunker = SemanticChunker(
    chunk_size=512,      # Target size in tokens
    chunk_overlap=50,    # Overlap for context
)

# Chunk your content
content = """
Your long document content here.
Multiple paragraphs, code blocks, etc.
"""

chunks = chunker.chunk_document(content)

# Access chunk information
for chunk in chunks:
    print(f"Chunk {chunk.chunk_index}: {chunk.token_count} tokens")
    print(chunk.content)
```

### 3. Chunk Knowledge Base Entries

```python
from rag.chunking import chunk_knowledge_base_entry

entry = {
    "id": "kb_001",
    "title": "Your Title",
    "content": "Long content...",
    "source": "Source Name",
    "year": 2024,
    "tags": ["tag1", "tag2"],
    "topic": "topic_name"
}

chunks = chunk_knowledge_base_entry(entry)
```

## Configuration Presets

### For Short Q&A
```python
chunker = SemanticChunker(
    chunk_size=256,
    chunk_overlap=30,
)
```

### For Technical Docs (Recommended)
```python
chunker = SemanticChunker(
    chunk_size=512,
    chunk_overlap=50,
)
```

### For Long-Form Content
```python
chunker = SemanticChunker(
    chunk_size=1024,
    chunk_overlap=100,
)
```

## Testing

```bash
# Run all tests
pytest tests/test_semantic_chunking.py -v

# Run examples
python examples_semantic_chunking.py
```

## Key Features

✅ **Preserves semantic boundaries** (paragraphs, sentences, code blocks)  
✅ **Token-aware** using tiktoken  
✅ **Configurable** chunk sizes and overlap  
✅ **Backward compatible** with existing code  
✅ **Well-tested** with 16 passing tests  

## Common Patterns

### Pattern 1: Custom Chunker
```python
from rag.chunking import SemanticChunker

chunker = SemanticChunker(
    chunk_size=400,
    chunk_overlap=40,
    min_chunk_size=80,
)
```

### Pattern 2: Disable Chunking
```python
from rag.retriever import RAGRetriever

# Use original behavior
retriever = RAGRetriever(use_chunking=False)
```

### Pattern 3: Process Multiple Documents
```python
from rag.chunking import SemanticChunker

chunker = SemanticChunker()

documents = [doc1, doc2, doc3]
all_chunks = []

for doc in documents:
    chunks = chunker.chunk_document(
        content=doc["content"],
        metadata={"source": doc["source"]},
        doc_id=doc["id"]
    )
    all_chunks.extend(chunks)
```

## Troubleshooting

### Issue: "tiktoken not found"
**Solution:** Install tiktoken: `pip install tiktoken`  
**Note:** System falls back to character estimation if tiktoken is unavailable

### Issue: Chunks too large/small
**Solution:** Adjust `chunk_size` parameter:
```python
chunker = SemanticChunker(chunk_size=YOUR_SIZE)
```

### Issue: Loss of context at boundaries
**Solution:** Increase `chunk_overlap`:
```python
chunker = SemanticChunker(chunk_overlap=100)
```

## More Information

- **Full Documentation:** `docs/SEMANTIC_CHUNKING.md`
- **Implementation Summary:** `SEMANTIC_CHUNKING_IMPLEMENTATION.md`
- **Examples:** `examples_semantic_chunking.py`
- **Tests:** `tests/test_semantic_chunking.py`

## Support

For issues or questions:
1. Check the full documentation
2. Run the example script
3. Review the test suite
4. Open an issue on the repository
