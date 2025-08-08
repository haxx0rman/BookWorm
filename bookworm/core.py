"""
Core components for BookWorm: DocumentProcessor, KnowledgeGraph, and MindmapGenerator
Updated architecture: Each document gets its own knowledge graph
"""
import asyncio
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import tempfile
import shutil

# Internal imports
from .utils import BookWormConfig, get_file_category, is_supported_file
from .library import LibraryManager, DocumentStatus, DocumentType
from .mindmap_generator import AdvancedMindmapGenerator

# Set up logger
logger = logging.getLogger(__name__)

# For PDF processing
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

# For document processing
try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

# LightRAG components
try:
    from lightrag import LightRAG, QueryParam
    from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
    from lightrag.kg.shared_storage import initialize_pipeline_status
    from lightrag.utils import setup_logger as lightrag_setup_logger
except ImportError:
    LightRAG = None
    QueryParam = None
    gpt_4o_mini_complete = None
    openai_embed = None
    initialize_pipeline_status = None
    lightrag_setup_logger = None


@dataclass
class ProcessedDocument:
    """Data class for processed documents"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    original_path: str = ""
    processed_path: str = ""
    text_content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    file_type: str = ""
    file_size: int = 0
    processed_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"  # pending, processing, completed, failed
    error_message: Optional[str] = None


@dataclass 
class MindmapResult:
    """Data class for mindmap generation results"""
    document_id: str
    mermaid_syntax: str = ""
    html_content: str = ""
    markdown_outline: str = ""
    generated_at: datetime = field(default_factory=datetime.now)
    provider: str = ""
    token_usage: Dict[str, int] = field(default_factory=dict)


class DocumentProcessor:
    """Process documents and extract knowledge"""
    
    def __init__(self, config: BookWormConfig, library_manager: Optional[LibraryManager] = None):
        self.config = config
        self.working_dir = Path(config.working_dir)
        self.logger = logging.getLogger(__name__)
        self.library_manager = library_manager or LibraryManager(config)
    
    async def process_document(self, file_path: Union[str, Path]) -> Optional[ProcessedDocument]:
        """Process a single document and extract text"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            self.logger.error(f"File not found: {file_path}")
            return None
        
        if not is_supported_file(file_path):
            self.logger.warning(f"Unsupported file type: {file_path}")
            return None
        
        self.logger.info(f"Processing document: {file_path.name}")
        
        # Create processed document
        doc = ProcessedDocument(
            original_path=str(file_path.absolute()),
            file_type=file_path.suffix.lower(),
            file_size=file_path.stat().st_size,
            status="processing"
        )
        
        try:
            # Extract text based on file type
            doc.text_content = await self._extract_text(file_path)
            
            if doc.text_content:
                # Save processed text
                processed_dir = Path(self.config.processed_dir)
                processed_dir.mkdir(parents=True, exist_ok=True)
                processed_file = processed_dir / f"{doc.id}_{file_path.stem}.txt"
                processed_file.write_text(doc.text_content, encoding='utf-8')
                doc.processed_path = str(processed_file)
                doc.status = "completed"
                
                self.logger.info(f"✅ Successfully processed: {file_path.name}")
                return doc
            else:
                doc.status = "failed"
                doc.error_message = "No text content extracted"
                self.logger.error(f"❌ No text content extracted from: {file_path.name}")
                return None
                
        except Exception as e:
            doc.status = "failed"
            doc.error_message = str(e)
            self.logger.error(f"❌ Error processing {file_path.name}: {e}")
            return None
    
    async def _extract_text(self, file_path: Path) -> str:
        """Extract text from various file formats"""
        file_ext = file_path.suffix.lower()
        
        if file_ext == '.txt':
            return file_path.read_text(encoding='utf-8', errors='ignore')
        
        elif file_ext == '.md':
            return file_path.read_text(encoding='utf-8', errors='ignore')
        
        elif file_ext == '.pdf':
            return await self._extract_pdf_text(file_path)
        
        elif file_ext in ['.doc', '.docx']:
            return await self._extract_word_text(file_path)
        
        else:
            # Try to read as text
            try:
                return file_path.read_text(encoding='utf-8', errors='ignore')
            except Exception as e:
                self.logger.warning(f"Could not read {file_path} as text: {e}")
                return ""
    
    async def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from PDF using multiple methods"""
        text_content = ""
        
        # Try PyMuPDF first
        if fitz:
            try:
                doc = fitz.open(str(file_path))
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    text_content += page.get_text() + "\n\n"
                doc.close()
                
                if text_content.strip():
                    self.logger.info(f"Extracted text using PyMuPDF: {len(text_content)} chars")
                    return text_content
            except Exception as e:
                self.logger.warning(f"PyMuPDF extraction failed: {e}")
        
        # Fallback to pdfplumber
        if pdfplumber:
            try:
                with pdfplumber.open(str(file_path)) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_content += page_text + "\n\n"
                
                if text_content.strip():
                    self.logger.info(f"Extracted text using pdfplumber: {len(text_content)} chars")
                    return text_content
            except Exception as e:
                self.logger.warning(f"pdfplumber extraction failed: {e}")
        
        self.logger.error("All PDF extraction methods failed")
        return ""
    
    async def _extract_word_text(self, file_path: Path) -> str:
        """Extract text from Word documents"""
        if not DocxDocument:
            self.logger.error("python-docx not available for Word document processing")
            return ""
        
        try:
            doc = DocxDocument(str(file_path))
            text_content = ""
            for paragraph in doc.paragraphs:
                text_content += paragraph.text + "\n"
            
            self.logger.info(f"Extracted text from Word document: {len(text_content)} chars")
            return text_content
            
        except Exception as e:
            self.logger.error(f"Word document extraction failed: {e}")
            return ""
    
    async def process_directory(self, directory_path: Union[str, Path]) -> List[ProcessedDocument]:
        """Process all supported documents in a directory"""
        directory_path = Path(directory_path)
        documents = []
        
        if not directory_path.exists():
            self.logger.error(f"Directory not found: {directory_path}")
            return documents
        
        self.logger.info(f"Processing documents in: {directory_path}")
        
        for file_path in directory_path.rglob("*"):
            if file_path.is_file() and is_supported_file(file_path):
                processed_doc = await self.process_document(file_path)
                if processed_doc:
                    documents.append(processed_doc)
        
        self.logger.info(f"Processed {len(documents)} documents from {directory_path}")
        return documents


class DocumentKnowledgeGraph:
    """Handles individual document knowledge graphs - one graph per document"""
    
    def __init__(self, config: BookWormConfig, document_id: str, library_manager: Optional[LibraryManager] = None):
        self.config = config
        self.document_id = document_id
        self.logger = logging.getLogger(f"bookworm.doc_kg.{document_id[:8]}")
        self.rag: Optional[LightRAG] = None
        self.library_manager = library_manager or LibraryManager(config)
        self.is_initialized = False
        
        # Create document-specific working directory in knowledge_graphs folder
        self.doc_working_dir = Path("./bookworm_workspace/knowledge_graphs") / document_id
        self.doc_working_dir.mkdir(parents=True, exist_ok=True)
        
        if not LightRAG:
            raise ImportError("LightRAG not available. Install lightrag-hku.")
    
    async def initialize(self) -> None:
        """Initialize document-specific LightRAG system"""
        if self.is_initialized:
            self.logger.info(f"Document KG {self.document_id[:8]} already initialized")
            return
            
        self.logger.info(f"Initializing document knowledge graph for {self.document_id[:8]}...")
        
        # Try to import Ollama functions (following user's pattern), fallback to OpenAI
        ollama_available = False
        try:
            from lightrag.llm.ollama import ollama_model_complete
            from lightrag.utils import EmbeddingFunc
            ollama_available = True
            self.logger.info("Using Ollama LLM functions")
        except ImportError:
            self.logger.warning("Ollama LLM functions not available, using OpenAI")
            try:
                from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
                ollama_available = False
            except ImportError:
                self.logger.error("Neither Ollama nor OpenAI LLM functions available")
                raise ImportError("No LLM functions available")
            
        # Debug configuration
        self.logger.info("Configuration:")
        self.logger.info(f"  Working Directory: {self.doc_working_dir}")
        self.logger.info(f"  LLM Model: {self.config.llm_model}")
        self.logger.info(f"  Embedding Model: {self.config.embedding_model}")
        
        # Initialize LightRAG with document-specific working directory
        if ollama_available:
            # Use Ollama (following user's pattern)
            self.rag = LightRAG(
                working_dir=str(self.doc_working_dir),
                llm_model_func=ollama_model_complete,
                llm_model_name=self.config.llm_model,
                llm_model_kwargs={
                    "host": self.config.llm_host,
                    "options": {"num_ctx": 18192, "num_threads": 11},
                    "timeout": self.config.timeout,
                },
                embedding_func=EmbeddingFunc(
                    embedding_dim=self.config.embedding_dim,
                    max_token_size=self.config.max_embed_tokens,
                    func=lambda texts: self._robust_ollama_embed(
                        texts,
                        embed_model=self.config.embedding_model,
                        host=self.config.embedding_host
                    ),
                ),
                vector_storage="FaissVectorDBStorage",
                cosine_better_than_threshold=0.3,
            )
        else:
            # Fallback to OpenAI
            self.logger.info("Using OpenAI as LLM provider")
            self.rag = LightRAG(
                working_dir=str(self.doc_working_dir),
                embedding_func=openai_embed,
                llm_model_func=gpt_4o_mini_complete,
            )
        
        # Initialize storage backends and pipeline
        await self.rag.initialize_storages()
        if initialize_pipeline_status:
            await initialize_pipeline_status()
        
        self.is_initialized = True
        self.logger.info(f"Document knowledge graph {self.document_id[:8]} initialized successfully")
    
    async def _robust_ollama_embed(self, texts, embed_model, host, max_retries=3, delay=1):
        """Robust wrapper for ollama_embed with retry logic"""
        from lightrag.llm.ollama import ollama_embed
        
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"Embedding processing attempt {attempt + 1} with model {embed_model}")
                return await ollama_embed(
                    texts, 
                    embed_model=embed_model, 
                    host=host,
                    timeout=self.config.embedding_timeout,
                    options={"num_threads": 11}
                )
            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"Embedding failed after {max_retries} attempts: {e}")
                    raise
                
                self.logger.warning(f"Embedding attempt {attempt + 1} failed: {e}")
                self.logger.info(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
                delay *= 2  # Exponential backoff
    
    async def add_document_content(self, content: str) -> None:
        """Add document content to this specific knowledge graph"""
        if not self.rag:
            raise RuntimeError("Knowledge graph not initialized. Call initialize() first.")
        
        if not content:
            raise ValueError("Document has no content")
        
        try:
            # Insert document content into this document's LightRAG
            self.logger.info(f"Adding content to document KG {self.document_id[:8]} (this may take a while)...")
            await self.rag.ainsert(content)
            self.logger.info(f"✅ Successfully added content to document KG {self.document_id[:8]}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to add content to document KG {self.document_id[:8]}: {e}")
            raise
    
    async def query(self, query: str, mode: str = "hybrid", stream: bool = False, **kwargs) -> str:
        """Query this document's knowledge graph"""
        if not self.is_initialized:
            await self.initialize()
        
        # Valid modes from user's lightrag_manager.py
        valid_modes = ["local", "global", "hybrid", "naive", "mix", "bypass"]
        if mode not in valid_modes:
            self.logger.warning(f"Invalid mode '{mode}'. Using 'hybrid' instead.")
            mode = "hybrid"
            
        try:
            self.logger.info(f"Querying document KG {self.document_id[:8]} in {mode} mode: {query}")
            
            if stream:
                # Handle streaming response
                result_stream = await self.rag.aquery(
                    query,
                    param=QueryParam(mode=mode, stream=stream, **kwargs),
                )
                result = ""
                async for chunk in result_stream:
                    result += chunk
                return result
            else:
                # Handle non-streaming response
                result = await self.rag.aquery(
                    query,
                    param=QueryParam(mode=mode, stream=stream, **kwargs),
                )
                return result
                
        except Exception as e:
            self.logger.error(f"Query failed on document KG {self.document_id[:8]}: {e}")
            raise


class KnowledgeGraph:
    """Manages multiple document knowledge graphs - one graph per document"""
    
    def __init__(self, config: BookWormConfig, library_manager: Optional[LibraryManager] = None):
        self.config = config
        self.logger = logging.getLogger("bookworm.knowledge_graph_manager")
        self.library_manager = library_manager or LibraryManager(config)
        self.document_graphs: Dict[str, DocumentKnowledgeGraph] = {}
        
        if not LightRAG:
            raise ImportError("LightRAG not available. Install lightrag-hku.")
    
    async def create_document_graph(self, document: ProcessedDocument) -> Tuple[DocumentKnowledgeGraph, Optional[str]]:
        """Create a new knowledge graph for a specific document"""
        self.logger.info(f"Creating knowledge graph for document {document.id[:8]}...")
        
        # Create document-specific knowledge graph
        doc_kg = DocumentKnowledgeGraph(self.config, document.id, self.library_manager)
        await doc_kg.initialize()
        
        # Add document content to the graph
        await doc_kg.add_document_content(document.text_content)
        
        # Store reference to the graph
        self.document_graphs[document.id] = doc_kg
        
        # Add to library manager
        library_doc_id = None
        try:
            if Path(document.original_path).exists():
                # Check if document already exists in library
                existing_docs = self.library_manager.find_documents(filename=Path(document.original_path).name)
                if not existing_docs:
                    library_doc_id = self.library_manager.add_document(filepath=document.original_path)
                    self.logger.info(f"📚 Document {library_doc_id} added to library index")
                else:
                    # Update existing document status
                    existing_doc = existing_docs[0]
                    library_doc_id = existing_doc.id
                    self.logger.info(f"📚 Found existing document {existing_doc.id}")
                
                # Always update the status to processed and knowledge graph path for both new and existing documents
                if library_doc_id:
                    self.library_manager.update_document_status(
                        library_doc_id, 
                        DocumentStatus.PROCESSED,
                        processed_file_path=document.processed_path
                    )
                    self.library_manager.update_document_metadata(
                        library_doc_id, 
                        {"knowledge_graph_id": str(doc_kg.doc_working_dir)}
                    )
                    self.logger.info(f"📚 Document {library_doc_id} status updated to PROCESSED")
                    self.logger.info(f"📚 Knowledge graph path saved to library: {doc_kg.doc_working_dir}")
            else:
                self.logger.warning(f"Original file {document.original_path} not found, skipping library update")
        except Exception as e:
            self.logger.warning(f"Failed to update library for document {document.id}: {e}")
        
        return doc_kg, library_doc_id
    
    async def get_document_graph(self, document_id: str) -> Optional[DocumentKnowledgeGraph]:
        """Get the knowledge graph for a specific document"""
        if document_id in self.document_graphs:
            return self.document_graphs[document_id]
        
        # Try to load from disk if not in memory
        doc_working_dir = Path("./bookworm_workspace/knowledge_graphs") / document_id
        if doc_working_dir.exists():
            self.logger.info(f"Loading existing knowledge graph for document {document_id[:8]}...")
            doc_kg = DocumentKnowledgeGraph(self.config, document_id, self.library_manager)
            await doc_kg.initialize()
            self.document_graphs[document_id] = doc_kg
            return doc_kg
        
        return None
    
    async def query_document(self, document_id: str, query: str, mode: str = "hybrid", **kwargs) -> Optional[str]:
        """Query a specific document's knowledge graph"""
        doc_kg = await self.get_document_graph(document_id)
        if doc_kg:
            return await doc_kg.query(query, mode=mode, **kwargs)
        return None
    
    async def query_all_documents(self, query: str, mode: str = "hybrid", **kwargs) -> Dict[str, str]:
        """Query all document knowledge graphs and return results"""
        results = {}
        
        # Load all existing graphs
        graphs_dir = Path("./bookworm_workspace/knowledge_graphs")
        if graphs_dir.exists():
            for doc_dir in graphs_dir.iterdir():
                if doc_dir.is_dir():
                    document_id = doc_dir.name
                    try:
                        doc_kg = await self.get_document_graph(document_id)
                        if doc_kg:
                            result = await doc_kg.query(query, mode=mode, **kwargs)
                            results[document_id] = result
                    except Exception as e:
                        self.logger.warning(f"Failed to query document {document_id[:8]}: {e}")
        
        return results
    
    def list_document_graphs(self) -> List[str]:
        """List all available document knowledge graphs"""
        graphs = []
        graphs_dir = Path("./bookworm_workspace/knowledge_graphs")
        if graphs_dir.exists():
            for doc_dir in graphs_dir.iterdir():
                if doc_dir.is_dir():
                    graphs.append(doc_dir.name)
        return graphs


class MindmapGenerator:
    """Generate mindmaps from knowledge graphs"""
    
    def __init__(self, config: BookWormConfig, library_manager: Optional[LibraryManager] = None):
        self.config = config
        self.working_dir = Path(config.working_dir)
        self.logger = logging.getLogger(__name__)
        self.library_manager = library_manager or LibraryManager(config)
        
        # Initialize the advanced mindmap generator
        self.advanced_generator = AdvancedMindmapGenerator(config)
    
    async def generate_mindmap(self, document: ProcessedDocument, document_library_id: Optional[str] = None) -> MindmapResult:
        """Generate a mindmap for a processed document using the advanced generator"""
        if not document.text_content:
            raise ValueError("Document has no text content")
        
        self.logger.info(f"🚀 Generating advanced mindmap for document {document.id}")
        
        try:
            # Use the advanced mindmap generator
            advanced_result = await self.advanced_generator.generate_mindmap_from_text(
                text_content=document.text_content,
                document_id=document.id
            )
            
            # Convert to our format
            result = MindmapResult(
                document_id=document.id,
                provider=self.config.api_provider,
                mermaid_syntax=advanced_result.mermaid_syntax,
                html_content=advanced_result.html_content,
                markdown_outline=advanced_result.markdown_outline,
                token_usage={
                    'input_tokens': advanced_result.token_usage.input_tokens,
                    'output_tokens': advanced_result.token_usage.output_tokens,
                    'total_tokens': advanced_result.token_usage.input_tokens + advanced_result.token_usage.output_tokens
                }
            )
            
            # Save results to files
            await self._save_mindmap_files(result, document, document_library_id)
            
            self.logger.info(f"✅ Advanced mindmap generated successfully for document {document.id}")
            self.logger.info(f"📊 Token usage: {result.token_usage['total_tokens']:,} total tokens")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate mindmap for document {document.id}: {e}")
            raise
    
    async def _save_mindmap_files(self, result: MindmapResult, document: ProcessedDocument, document_library_id: Optional[str] = None) -> None:
        """Save mindmap files and update library"""
        doc_name = Path(document.original_path).stem
        output_dir = Path("./bookworm_workspace/mindmaps") / result.document_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        mindmap_files = {}
        
        # Save Mermaid syntax
        if result.mermaid_syntax:
            mermaid_file = output_dir / f"{doc_name}_mindmap.mmd"
            mermaid_file.write_text(result.mermaid_syntax)
            mindmap_files["mermaid"] = str(mermaid_file)
            self.logger.info(f"💾 Saved Mermaid: {mermaid_file}")
        
        # Save HTML content
        if result.html_content:
            html_file = output_dir / f"{doc_name}_mindmap.html"
            html_file.write_text(result.html_content)
            mindmap_files["html"] = str(html_file)
            self.logger.info(f"💾 Saved HTML: {html_file}")
        
        # Save Markdown outline
        if result.markdown_outline:
            md_file = output_dir / f"{doc_name}_outline.md"
            md_file.write_text(result.markdown_outline)
            mindmap_files["markdown"] = str(md_file)
            self.logger.info(f"💾 Saved Markdown: {md_file}")
        
        # Add mindmap to library
        try:
            # Reload library data to ensure we have the latest state
            self.library_manager._load_library_state()
            
            if document_library_id:
                # Use the provided document library ID directly
                doc_id = document_library_id
                self.logger.info(f"Using provided document ID: {doc_id}")
            else:
                # Find the document in library by filename (fallback)
                existing_docs = self.library_manager.find_documents(filename=Path(document.original_path).name)
                if existing_docs:
                    doc_id = existing_docs[0].id
                    self.logger.info(f"Found document by filename: {doc_id}")
                else:
                    self.logger.error(f"Could not find document in library: {Path(document.original_path).name}")
                    return
            
            mindmap_metadata = {
                'document_type': 'unknown',
                'token_usage': result.token_usage.get('total_tokens', 0),
                'processing_time': 0.0,
                'generator_version': '1.0',
                'topic_count': 0,
                'subtopic_count': 0,
                'detail_count': 0
            }
            mindmap_id = self.library_manager.add_mindmap(
                document_id=doc_id,
                mindmap_files=mindmap_files,
                metadata=mindmap_metadata
            )
            self.logger.info(f"📚 Mindmap {mindmap_id} added to library")
            
            # Update the document record with the mindmap ID
            self.library_manager.update_document_metadata(
                doc_id, 
                {"mindmap_id": mindmap_id}
            )
            self.logger.info(f"📚 Document {doc_id} updated with mindmap ID: {mindmap_id}")
        except Exception as e:
            self.logger.error(f"Failed to add mindmap to library: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
