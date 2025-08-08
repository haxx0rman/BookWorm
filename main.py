#!/usr/bin/env python3
"""
BookWorm Demo Script - Following user's methodology

This demo shows how to use BookWorm with your existing LightRAG setup:
- Uses mineru for PDF processing (like lightrag_ex.py)
- Follows LightRAG query patterns (like lightrag_manager.py)
- Demonstrates the full document → knowledge graph → mindmap pipeline
"""
import asyncio
import logging
from pathlib import Path

from bookworm.utils import BookWormConfig, load_config, setup_logging
from bookworm.core import DocumentProcessor, KnowledgeGraph


async def demo_with_existing_files():
    """Demo using your existing methodology from lightrag_ex.py and lightrag_manager.py"""
    print("🐛 BookWorm Demo - Following your methodology")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    setup_logging(config)
    
    # Create components
    processor = DocumentProcessor(config)
    knowledge_graph = KnowledgeGraph(config)
    
    print("📁 Looking for documents to process...")
    
    # Look for documents in the workspace docs directory
    docs_to_process = []
    workspace_path = Path(config.working_dir)
    docs_dir = workspace_path / "docs"
    
    if docs_dir.exists():
        for doc_file in docs_dir.rglob("*"):
            if doc_file.is_file() and doc_file.suffix.lower() in ['.pdf', '.txt', '.md', '.docx']:
                docs_to_process.append(doc_file)
                print(f"  📄 Found document: {doc_file.name}")
    
    # Check for already processed documents
    processed_docs_dir = workspace_path / "processed_docs"
    if processed_docs_dir.exists():
        for doc_file in processed_docs_dir.rglob("*"):
            if doc_file.is_file() and doc_file.suffix.lower() in ['.txt', '.md']:
                print(f"  📝 Found processed: {doc_file.name}")
    
    if not docs_to_process:
        print("  ℹ️  No documents found in docs directory. Creating sample documents...")
        await create_sample_documents(config)
        return
    
    print("🔧 Using your configuration:")
    print(f"  PDF Processor: {config.pdf_processor} (following your mineru methodology)")
    print(f"  LLM Model: {config.llm_model}")
    print(f"  Embedding Model: {config.embedding_model}")
    print(f"  Working Directory: {config.working_dir}")
    
    # Initialize knowledge graph
    print("🧠 Initializing Knowledge Graph (following your LightRAG setup)...")
    await knowledge_graph.initialize()
    
    # Process documents from docs directory
    if docs_to_process:
        print(f"📚 Processing {len(docs_to_process)} documents...")
        
        all_processed = []
        for file_path in docs_to_process:
            try:
                print(f"  📄 Processing: {file_path.name}")
                
                # Step 1: Extract text from document
                processed_doc = await processor.process_document(file_path)
                
                if processed_doc and processed_doc.text_content:
                    print(f"    📝 Text extracted: {len(processed_doc.text_content)} chars")
                    all_processed.append(processed_doc)
                    
                    # Step 2: Add to knowledge graph (this is the slow part)
                    print(f"    🧠 Adding to knowledge graph (this may take a while)...")
                    await knowledge_graph.add_document(processed_doc)
                    print(f"    ✅ Completed: {file_path.name}")
                else:
                    print(f"    ❌ Failed to extract text from: {file_path.name}")
                    
            except Exception as e:
                print(f"    ❌ Error processing {file_path.name}: {e}")
        
        print(f"🎉 Successfully processed {len(all_processed)} documents into knowledge graph")
        
        # Query examples following your lightrag_manager.py patterns
        print("🔍 Running Knowledge Graph Queries (following your methodology)...")
        
        query_examples = [
            ("What is LightRAG and how does it work?", "hybrid"),
            ("How is PDF processing implemented?", "local"),
            ("What are the main components of the system?", "global"),
            ("Explain the mineru PDF processing approach", "hybrid"),
        ]
        
        for query, mode in query_examples:
            try:
                print(f"  🔍 Query ({mode}): {query}")
                result = await knowledge_graph.query(query, mode=mode)
                print(f"    📝 Result: {result[:200]}...")
                
            except Exception as e:
                print(f"    ❌ Query failed: {e}")
    
    print("🎉 Demo completed! Your BookWorm system is working with your methodology.")
    print(f"📁 Check outputs in: {config.output_dir}")
    print("💡 Next steps:")
    print("  1. Copy .env.example to .env and configure your API keys")
    print("  2. Add more documents to process")
    print("  3. Use 'uv run bookworm --help' for CLI commands")


async def create_sample_documents(config: BookWormConfig):
    """Create sample documents for demo if none exist"""
    print("📝 Creating sample documents...")
    
    # Create directories
    docs_dir = Path(config.document_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample document about BookWorm
    sample_doc = docs_dir / "bookworm_overview.md"
    sample_content = """# BookWorm - Document Knowledge System

BookWorm is an advanced document ingestion system that combines LightRAG for knowledge graph construction with intelligent mindmap generation.

## Key Features

### PDF Processing with MinerU
- Uses MinerU as the primary PDF processor
- Follows CPU-only processing for compatibility
- Generates high-quality markdown from PDFs
- Supports tables, formulas, and complex layouts

### LightRAG Integration
- Builds knowledge graphs from document content
- Supports multiple query modes: local, global, hybrid, naive, mix
- Uses Ollama for local LLM processing
- Maintains persistent knowledge across sessions

### Mindmap Generation
- Creates hierarchical visualizations of document content
- Supports multiple output formats (Mermaid, HTML, Markdown)
- Uses LLM-powered analysis for content structuring
- Adapts to different document types and domains

## Technical Architecture

The system follows a pipeline approach:
1. Document ingestion and text extraction
2. Knowledge graph construction with LightRAG
3. Query processing and retrieval
4. Mindmap generation and visualization

This approach enables comprehensive document analysis and knowledge discovery.
"""
    
    sample_doc.write_text(sample_content)
    print(f"  ✅ Created: {sample_doc}")
    
    # Process the sample document
    processor = DocumentProcessor(config)
    knowledge_graph = KnowledgeGraph(config)
    
    await knowledge_graph.initialize()
    
    processed_doc = await processor.process_document(sample_doc)
    if processed_doc:
        await knowledge_graph.add_document(processed_doc)
        print("  ✅ Added sample document to knowledge graph")


if __name__ == "__main__":
    asyncio.run(demo_with_existing_files())
