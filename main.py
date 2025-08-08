#!/usr/bin/env python3
"""
BookWorm Document Processing System

Each document creates its own knowledge graph for better isolation and scalability.
No hardcoded sample data - processes real documents only.
Now supports directory processing for Obsidian vaults and document collections.
"""
import asyncio
import logging
from pathlib import Path

from bookworm.utils import BookWormConfig, load_config, setup_logging
from bookworm.core import DocumentProcessor, KnowledgeGraph, MindmapGenerator
from bookworm.library import LibraryManager, DocumentStatus, DocumentType


async def process_directory_collection(processor, knowledge_graph, mindmap_generator, library_manager, directory_path):
    """Process an entire directory as a single document"""
    print(f"📁 Processing directory collection: {directory_path.name}")
    
    try:
        # Check if directory is already processed in library
        existing_doc = library_manager.get_document_by_filename(directory_path.name)
        if existing_doc and existing_doc.status == DocumentStatus.PROCESSED:
            print(f"    ⏭️  Directory already processed, skipping: {directory_path.name}")
            return None
        
        # Process directory as single document
        processed_doc = await processor.process_directory_as_single_document(directory_path)
        
        if processed_doc:
            print(f"    📝 Combined text extracted: {len(processed_doc.text_content):,} characters")
            print(f"    📄 Files processed: {processed_doc.metadata.get('file_count', 0)}")
            
            # Create individual knowledge graph for this directory collection
            print("    🧠 Creating knowledge graph...")
            doc_kg, library_doc_id = await knowledge_graph.create_document_graph(processed_doc)
            print(f"    ✅ Knowledge graph created: {processed_doc.id[:8]}")
            
            # Generate mindmap
            print("    🗺️  Generating mindmap visualization...")
            try:
                mindmap_result = await mindmap_generator.generate_mindmap(processed_doc, library_doc_id)
                print(f"    ✅ Mindmap generated (tokens: {mindmap_result.token_usage.get('total_tokens', 0):,})")
            except Exception as mindmap_error:
                print(f"    ⚠️  Mindmap generation failed: {mindmap_error}")
            
            print(f"    ✅ Completed directory: {directory_path.name}")
            return doc_kg
        else:
            print(f"    ❌ Failed to process directory: {directory_path.name}")
            return None
            
    except Exception as e:
        print(f"    ❌ Error processing directory {directory_path.name}: {e}")
        return None


async def process_documents():
    """Process documents found in the workspace - no hardcoded samples"""
    print("🐛 BookWorm Document Processing System")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    setup_logging(config)
    
    # Create components with shared library manager
    library_manager = LibraryManager(config)
    processor = DocumentProcessor(config, library_manager)  # Pass shared library manager
    knowledge_graph = KnowledgeGraph(config, library_manager)  # Pass shared library manager
    mindmap_generator = MindmapGenerator(config, library_manager)  # Pass shared library manager
    
    print("📚 Library Status:")
    stats = library_manager.get_library_stats()
    print(f"  📄 Documents: {stats.total_documents}")
    print(f"  🗺️  Mindmaps: {stats.total_mindmaps}")
    print(f"  ✅ Processed: {stats.processed_documents}")
    print(f"  ⏳ Pending: {stats.pending_documents}")
    print(f"  ❌ Failed: {stats.failed_documents}")
    print(f"  💾 Total size: {stats.total_size_bytes:,} bytes")
    print()
    
    print("📁 Scanning for documents and directories to process...")
    
    # Look for documents and directories in the workspace docs directory
    docs_to_process = []
    dirs_to_process = []
    workspace_path = Path(config.working_dir)
    docs_dir = workspace_path / "docs"
    
    # Only check the docs directory to avoid duplicates
    if docs_dir.exists():
        for item_path in docs_dir.iterdir():
            if item_path.is_file() and item_path.suffix.lower() in ['.txt', '.md', '.pdf', '.docx']:
                # Skip if it's in processed directories
                if 'processed' not in str(item_path) and 'output' not in str(item_path):
                    docs_to_process.append(item_path)
            elif item_path.is_dir() and not item_path.name.startswith('.'):
                # Check if directory contains supported files
                has_supported_files = any(
                    f.is_file() and f.suffix.lower() in ['.txt', '.md', '.pdf', '.docx']
                    for f in item_path.rglob("*")
                )
                if has_supported_files:
                    dirs_to_process.append(item_path)
    
    if not docs_to_process and not dirs_to_process:
        print("📝 No documents or directories found in workspace.")
        print(f"   Add documents/directories to: {docs_dir}")
        print("   Supported formats: .txt, .md, .pdf, .docx")
        print("   Directories: Obsidian vaults, document collections")
        return
    
    if docs_to_process:
        print(f"📄 Found {len(docs_to_process)} individual documents:")
        for doc_path in docs_to_process:
            print(f"  • {doc_path.name}")
    
    if dirs_to_process:
        print(f"📁 Found {len(dirs_to_process)} directories to process as collections:")
        for dir_path in dirs_to_process:
            # Count files in directory
            file_count = sum(1 for f in dir_path.rglob("*") 
                           if f.is_file() and f.suffix.lower() in ['.txt', '.md', '.pdf', '.docx'])
            print(f"  • {dir_path.name} ({file_count} files)")
    
    print()
    
    # Process documents and directories
    document_graphs = {}
    
    # Process individual documents first
    for file_path in docs_to_process:
        print(f"🔄 Processing document: {file_path.name}")
        
        try:
            # Check if document is already processed in library
            existing_doc = library_manager.get_document_by_filename(file_path.name)
            if existing_doc and existing_doc.status == DocumentStatus.PROCESSED:
                print(f"    ⏭️  Already processed, skipping: {file_path.name}")
                continue
            
            # Extract text from document
            processed_doc = await processor.process_document(file_path)
            
            if processed_doc:
                print(f"    📝 Text extracted: {len(processed_doc.text_content):,} characters")
                
                # Create individual knowledge graph for this document
                print("    🧠 Creating knowledge graph...")
                doc_kg, library_doc_id = await knowledge_graph.create_document_graph(processed_doc)
                document_graphs[processed_doc.id] = doc_kg
                print(f"    ✅ Knowledge graph created: {processed_doc.id[:8]}")
                
                # Generate mindmap
                print("    🗺️  Generating mindmap visualization...")
                try:
                    mindmap_result = await mindmap_generator.generate_mindmap(processed_doc, library_doc_id)
                    print(f"    ✅ Mindmap generated (tokens: {mindmap_result.token_usage.get('total_tokens', 0):,})")
                except Exception as mindmap_error:
                    print(f"    ⚠️  Mindmap generation failed: {mindmap_error}")
                
                print(f"    ✅ Completed: {file_path.name}")
            else:
                print(f"    ❌ Failed to extract text from: {file_path.name}")
                
        except Exception as e:
            print(f"    ❌ Error processing {file_path.name}: {e}")
    
    # Process directories as collections
    for dir_path in dirs_to_process:
        doc_kg = await process_directory_collection(
            processor, knowledge_graph, mindmap_generator, library_manager, dir_path
        )
        if doc_kg:
            document_graphs[f"dir_{dir_path.name}"] = doc_kg
    
    if document_graphs:
        print(f"\n🎉 Successfully processed {len(document_graphs)} documents")
        print("   Each document has its own isolated knowledge graph")
        
        # Demonstrate querying individual documents
        print("\n🔍 Testing Knowledge Graph Queries...")
        
        # Sample queries to test on each document
        test_queries = [
            "What is the main topic of this document?",
            "What are the key concepts discussed?",
            "Summarize the main points"
        ]
        
        for doc_id, doc_kg in list(document_graphs.items())[:2]:  # Test first 2 documents
            print(f"\n  📄 Document: {doc_id[:8]}")
            
            for query in test_queries[:1]:  # Test first query only to avoid too much output
                try:
                    result = await doc_kg.query(query, mode="hybrid")
                    print(f"    Q: {query}")
                    print(f"    A: {result[:150]}...")
                    break  # Only show one query per document
                except Exception as e:
                    print(f"    ❌ Query failed: {e}")
        
        # Show cross-document capabilities
        print("\n🔄 Cross-Document Query Example:")
        print(f"   Querying all {len(document_graphs)} document graphs...")
        
        try:
            cross_results = await knowledge_graph.query_all_documents(
                "What are the main topics across all documents?", 
                mode="global"
            )
            print(f"   ✅ Received results from {len(cross_results)} documents")
            for doc_id, result in list(cross_results.items())[:1]:  # Show one example
                print(f"   {doc_id[:8]}: {result[:100]}...")
        except Exception as e:
            print(f"   ⚠️  Cross-document query failed: {e}")
    
    # Display final library statistics
    print("\n📊 Final Library Statistics:")
    final_stats = library_manager.get_library_stats()
    print(f"  📄 Total Documents: {final_stats.total_documents}")
    print(f"  🗺️  Total Mindmaps: {final_stats.total_mindmaps}")
    print(f"  ✅ Successfully Processed: {final_stats.processed_documents}")
    print(f"  💾 Total Content Size: {final_stats.total_size_bytes:,} bytes")
    print(f"  🕒 Last Updated: {final_stats.last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Show knowledge graph directories
    kg_list = knowledge_graph.list_document_graphs()
    print("\n🧠 Knowledge Graph Status:")
    print(f"  📊 Individual graphs created: {len(kg_list)}")
    print("  📁 Graph storage: ./bookworm_workspace/knowledge_graphs/")
    
    print("\n🎉 Processing completed!")
    print(f"📁 Check outputs in: {config.output_dir}")
    print("💡 Architecture:")
    print("  • Each document has its own isolated knowledge graph")
    print("  • All graphs are indexed in the library system")
    print("  • Cross-document queries are supported")
    print("  • Mindmaps are generated per document")


if __name__ == "__main__":
    asyncio.run(process_documents())
