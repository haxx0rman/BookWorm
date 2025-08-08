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
    
    print("📁 Examining your existing demo files...")
    
    # Look for your demo files
    demo_files = []
    current_dir = Path(".")
    
    # Your script files
    script_files = ["lightrag_ex.py", "lightrag_manager.py"]
    for script_file in script_files:
        if (current_dir / script_file).exists():
            demo_files.append(current_dir / script_file)
            print(f"  ✅ Found: {script_file}")
    
    # Look for any existing workspace
    workspace_dirs = ["./lightrag_workspace", "./bookworm_workspace"]
    existing_docs = []
    
    for workspace_dir in workspace_dirs:
        workspace_path = Path(workspace_dir)
        if workspace_path.exists():
            docs_dir = workspace_path / "docs"
            processed_docs_dir = workspace_path / "processed_docs"
            
            # Check for documents in docs directory
            if docs_dir.exists():
                for doc_file in docs_dir.rglob("*"):
                    if doc_file.is_file() and doc_file.suffix.lower() in ['.pdf', '.txt', '.md', '.docx']:
                        existing_docs.append(doc_file)
                        print(f"  📄 Found document: {doc_file}")
            
            # Check for processed documents
            if processed_docs_dir.exists():
                for doc_file in processed_docs_dir.rglob("*"):
                    if doc_file.is_file() and doc_file.suffix.lower() in ['.txt', '.md']:
                        existing_docs.append(doc_file)
                        print(f"  📝 Found processed: {doc_file}")
    
    if not demo_files and not existing_docs:
        print("  ℹ️  No demo files found. Creating sample documents...")
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
    
    # Process documents if found
    if demo_files or existing_docs:
        print(f"📚 Processing {len(demo_files + existing_docs)} documents...")
        
        all_processed = []
        for file_path in demo_files + existing_docs:
            try:
                print(f"  Processing: {file_path.name}")
                
                # Process the document
                processed_doc = await processor.process_document(file_path)
                
                if processed_doc and processed_doc.text_content:
                    all_processed.append(processed_doc)
                    
                    # Add to knowledge graph
                    await knowledge_graph.add_document(processed_doc)
                    print(f"    ✅ Added to knowledge graph: {len(processed_doc.text_content)} chars")
                else:
                    print(f"    ❌ Failed to process: {file_path.name}")
                    
            except Exception as e:
                print(f"    ❌ Error processing {file_path.name}: {e}")
        
        print(f"✅ Successfully processed {len(all_processed)} documents")
        
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
