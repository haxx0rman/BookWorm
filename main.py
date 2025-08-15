#!/usr/bin/env python3
"""
BookWorm Document Processing System

Each document creates its own knowledge graph for better isolation and scalability.
No hardcoded sample data - processes real documents only.
Now supports directory processing for Obsidian vaults and document collections.

Updated to use the new modular architecture:
- processors: DocumentProcessor
- knowledge: KnowledgeGraph  
- generators: MindmapGenerator
"""
import asyncio
import logging
from pathlib import Path
import os
import shutil
import tempfile

# PDF conversion imports will be loaded dynamically as needed

from bookworm.utils import BookWormConfig, load_config, setup_logging
from bookworm.processors import DocumentProcessor
from bookworm.knowledge import KnowledgeGraph
from bookworm.generators import MindmapGenerator
from bookworm.library import LibraryManager, DocumentStatus, DocumentType

from dotenv import load_dotenv
load_dotenv(dotenv_path=".env", override=False)

import logging
logging.basicConfig(level=logging.INFO)


async def convert_and_archive_pdfs(config: BookWormConfig):
    """
    Convert all PDFs in the document directory to markdown using the configured processor,
    and archive the original PDFs to the processed_docs directory.
    """
    docs_dir = Path(config.document_dir)
    processed_dir = Path(config.processed_dir)
    pdf_processor = config.pdf_processor.lower()
    os.makedirs(docs_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    pdf_files = list(docs_dir.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found to convert.")
        return

    print(f"\n📄 Converting {len(pdf_files)} PDF files to Markdown using {pdf_processor.title()}...")
    for i, pdf_path in enumerate(pdf_files, 1):
        try:
            print(f"[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")
            markdown_filename = pdf_path.stem + ".md"
            markdown_path = docs_dir / markdown_filename
            pdf_archive_path = processed_dir / pdf_path.name

            # Skip if already converted and archived
            if markdown_path.exists() and pdf_archive_path.exists():
                print(f"✅ Already processed: {markdown_path} and archived PDF.")
                continue

            # Convert PDF to markdown
            markdown_content = ""
            if pdf_processor == "docling":
                try:
                    from docling.document_converter import DocumentConverter
                    from docling.datamodel.base_models import InputFormat
                    from docling.datamodel.pipeline_options import PdfPipelineOptions
                    from docling.document_converter import PdfFormatOption
                    print("Processing with Docling...")
                    pipeline_options = PdfPipelineOptions()
                    pipeline_options.do_ocr = True
                    pipeline_options.do_table_structure = True
                    pipeline_options.table_structure_options.do_cell_matching = True
                    doc_converter = DocumentConverter(
                        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
                    )
                    result = doc_converter.convert(str(pdf_path))
                    if result and result.document:
                        markdown_content = result.document.export_to_markdown()
                except ImportError as e:
                    print(f"Docling not available: {e}. Falling back to MinerU...")
                    pdf_processor = "mineru"  # fallback
                except Exception as e:
                    print(f"Error converting PDF with Docling: {e}")
            if pdf_processor == "mineru" and not markdown_content:
                try:
                    from mineru.cli.common import prepare_env, read_fn
                    from mineru.data.data_reader_writer import FileBasedDataWriter
                    from mineru.utils.enum_class import MakeMode
                    from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
                    from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
                    from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
                    import os as _os
                    _os.environ["CUDA_VISIBLE_DEVICES"] = ""
                    with tempfile.TemporaryDirectory() as temp_dir:
                        pdf_file_name = pdf_path.stem
                        local_image_dir, local_md_dir = prepare_env(temp_dir, pdf_file_name, "auto")
                        pdf_bytes = read_fn(str(pdf_path))
                        if not pdf_bytes:
                            print(f"Failed to read PDF file: {pdf_path}")
                        else:
                            pdf_bytes_list = [pdf_bytes]
                            p_lang_list = ["en"]
                            print("Processing with CPU-only mode...")
                            infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = pipeline_doc_analyze(
                                pdf_bytes_list, p_lang_list, parse_method="auto", formula_enable=True, table_enable=True
                            )
                            if infer_results:
                                model_list = infer_results[0]
                                images_list = all_image_lists[0]
                                pdf_doc = all_pdf_docs[0]
                                _lang = lang_list[0]
                                _ocr_enable = ocr_enabled_list[0]
                                image_writer = FileBasedDataWriter(local_image_dir)
                                middle_json = pipeline_result_to_middle_json(
                                    model_list, images_list, pdf_doc, image_writer, _lang, _ocr_enable, True
                                )
                                pdf_info = middle_json["pdf_info"]
                                image_dir = str(os.path.basename(local_image_dir))
                                markdown_content = pipeline_union_make(pdf_info, MakeMode.MM_MD, image_dir)
                                if isinstance(markdown_content, list):
                                    markdown_content = "\n".join(str(item) for item in markdown_content)
                except ImportError as e:
                    print(f"MinerU not available: {e}. Cannot convert PDF.")
                except Exception as e:
                    print(f"Error converting PDF with MinerU: {e}")
            if markdown_content:
                with open(markdown_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                print(f"✅ Converted to: {markdown_path} ({len(markdown_content)} characters)")
                try:
                    shutil.move(str(pdf_path), str(pdf_archive_path))
                    print(f"📦 Archived PDF: {pdf_archive_path}")
                except Exception as move_error:
                    print(f"⚠️  Warning: Could not archive PDF: {move_error}")
            else:
                print(f"❌ Failed to convert: {pdf_path}")
        except Exception as e:
            print(f"❌ Error converting {pdf_path}: {e}")


async def process_directory_collection(processor, knowledge_graph, mindmap_generator, library_manager, directory_path):
    """Process an entire directory as a single document"""
    print(f"\ud83d\udcc1 Processing directory collection: {directory_path.name}")
    
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
            
            # Generate AI description from knowledge graph
            print("    📄 Generating description from knowledge graph...")
            try:
                description = await doc_kg.generate_description()
                if description:
                    print(f"    ✅ Description generated: {description[:80]}...")
                else:
                    print("    ⚠️  Using fallback description")
                    description = f"Directory collection containing {processed_doc.metadata.get('file_count', 0)} files."
            except Exception as desc_error:
                print(f"    ⚠️  Description generation failed: {desc_error}")
                description = f"Directory collection containing {processed_doc.metadata.get('file_count', 0)} files."
            
            # Generate tags from knowledge graph
            print("    🏷️  Generating tags from knowledge graph...")
            try:
                tags = await doc_kg.generate_tags()
                if tags:
                    print(f"    ✅ Tags generated: {', '.join(tags[:3])}{'...' if len(tags) > 3 else ''}")
                else:
                    print("    ⚠️  Using fallback tags")
                    tags = ["directory", "collection"]
            except Exception as tags_error:
                print(f"    ⚠️  Tags generation failed: {tags_error}")
                tags = ["directory", "collection"]
            
            # Update library with description and tags
            if library_doc_id:
                try:
                    metadata_update = {}
                    if description:
                        metadata_update["description"] = description
                    if tags:
                        metadata_update["tags"] = tags
                    
                    if metadata_update:
                        library_manager.update_document_metadata(library_doc_id, metadata_update)
                        print("    📝 Description and tags saved to library")
                except Exception as e:
                    print(f"    ⚠️  Failed to save metadata: {e}")
            
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

    # Step 1: Convert and archive PDFs before processing anything else
    if not config.skip_pdf_conversion:
        await convert_and_archive_pdfs(config)
    else:
        print("\n⏭️  Skipping PDF conversion (SKIP_PDF_CONVERSION=true)")
    
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
                
                # Generate AI description from knowledge graph
                print("    📄 Generating description from knowledge graph...")
                try:
                    description = await doc_kg.generate_description()
                    if description:
                        print(f"    ✅ Description generated: {description[:80]}...")
                    else:
                        print("    ⚠️  Using fallback description")
                        description = f"Document processed: {file_path.name}"
                except Exception as desc_error:
                    print(f"    ⚠️  Description generation failed: {desc_error}")
                    description = f"Document processed: {file_path.name}"
                
                # Generate tags from knowledge graph
                print("    🏷️  Generating tags from knowledge graph...")
                try:
                    tags = await doc_kg.generate_tags()
                    if tags:
                        print(f"    ✅ Tags generated: {', '.join(tags[:3])}{'...' if len(tags) > 3 else ''}")
                    else:
                        print("    ⚠️  Using fallback tags")
                        tags = ["document"]
                except Exception as tags_error:
                    print(f"    ⚠️  Tags generation failed: {tags_error}")
                    tags = ["document"]
                
                # Update library with description and tags
                if library_doc_id:
                    try:
                        metadata_update = {}
                        if description:
                            metadata_update["description"] = description
                        if tags:
                            metadata_update["tags"] = tags
                        
                        if metadata_update:
                            library_manager.update_document_metadata(library_doc_id, metadata_update)
                            print("    📝 Description and tags saved to library")
                    except Exception as e:
                        print(f"    ⚠️  Failed to save metadata: {e}")
                
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

    # Interactive CLI for querying
    print("\n🖥️  Entering interactive query mode. Type 'help' for options.")
    while True:
        try:
            cmd = input("\nCommand (query/queryall/stats/list/exit/help): ").strip().lower()
            if cmd in ("exit", "quit"): 
                print("👋 Goodbye!")
                break
            elif cmd == "help":
                print("\nAvailable commands:")
                print("  query     - Query a specific document graph")
                print("  queryall  - Query all document graphs (cross-document)")
                print("  stats     - Show library/document stats")
                print("  list      - List available document graph IDs")
                print("  exit      - Exit interactive mode")
            elif cmd == "list":
                print("\nAvailable document graph IDs:")
                for doc_id in kg_list:
                    print(f"  - {doc_id}")
            elif cmd == "stats":
                stats = library_manager.get_library_stats()
                print(f"\n📊 Documents: {stats.total_documents}, Processed: {stats.processed_documents}, Mindmaps: {stats.total_mindmaps}, Failed: {stats.failed_documents}")
            elif cmd == "query":
                doc_id = input("Enter document graph ID: ").strip()
                if not doc_id:
                    print("No document ID entered.")
                    continue
                query = input("Enter your question: ").strip()
                if not query:
                    print("No query entered.")
                    continue
                try:
                    doc_kg = await knowledge_graph.get_document_graph(doc_id)
                    if not doc_kg:
                        print(f"No knowledge graph found for document ID: {doc_id}")
                        continue
                    answer = await doc_kg.query(query, mode="hybrid")
                    print(f"\nA: {answer}")
                except Exception as e:
                    print(f"Error querying document graph: {e}")
            elif cmd == "queryall":
                query = input("Enter your question for all documents: ").strip()
                if not query:
                    print("No query entered.")
                    continue
                try:
                    results = await knowledge_graph.query_all_documents(query, mode="global")
                    for doc_id, answer in results.items():
                        print(f"\n[Document {doc_id[:8]}]\n{answer}")
                except Exception as e:
                    print(f"Error in cross-document query: {e}")
            else:
                print("Unknown command. Type 'help' for options.")
        except Exception as e:
            print(f"Error in interactive CLI: {e}")

if __name__ == "__main__":
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    asyncio.run(process_documents())
