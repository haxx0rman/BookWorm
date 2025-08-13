"""
BookWorm Core Components - Compatibility Layer

This module provides backward compatibility by re-exporting the refactored components.
The original monolithic core.py has been broken down into focused modules:

- models/: Data models (ProcessedDocument, MindmapResult)
- processors/: Document processing and description generation  
- knowledge/: Knowledge graph management
- generators/: Content generation (mindmaps, etc.)
- library/: Document library management
- utils/: Configuration and utilities

For new code, import directly from the specific modules.
This file maintains compatibility for existing imports.
"""

# Re-export models
from .models import ProcessedDocument, MindmapResult

# Re-export processors  
from .processors import DocumentProcessor, DocumentDescriptionGenerator

# Re-export knowledge components
from .knowledge import DocumentKnowledgeGraph, KnowledgeGraph

# Re-export generators
from .generators import MindmapGenerator

# Maintain backward compatibility
__all__ = [
    'ProcessedDocument',
    'MindmapResult', 
    'DocumentProcessor',
    'DocumentDescriptionGenerator',
    'DocumentKnowledgeGraph',
    'KnowledgeGraph',
    'MindmapGenerator'
]
                            description = description[12:].strip()
                        
                        self.logger.info(f"📝 Generated description for document {document.id[:8]}: {description[:50]}...")
                        return description
                    else:
                        self.logger.warning(f"⚠️ Empty description generated for document {document.id}")
                        return self.generate_fallback_description(document)
                else:
                    self.logger.warning("⚠️ Ollama client not available, using fallback description")
                    return self.generate_fallback_description(document)
                    
            except Exception as e:
                self.logger.error(f"❌ Error calling Ollama for description generation: {e}")
                return self.generate_fallback_description(document)
                
        except Exception as e:
            self.logger.error(f"❌ Error generating description for document {document.id}: {e}")
            return self.generate_fallback_description(document)
    
    def generate_fallback_description(self, document: ProcessedDocument) -> str:
        """Generate a simple fallback description based on metadata"""
        try:
            doc_type = "Directory collection" if document.metadata.get("is_directory", False) else "Document"
            
            if document.metadata.get("is_directory", False):
                file_count = document.metadata.get("file_count", 0)
                return f"{doc_type} containing {file_count} files with {len(document.text_content):,} characters of combined content."
            else:
                word_count = len(document.text_content.split()) if document.text_content else 0
                return f"{doc_type} with approximately {word_count:,} words covering various topics."
                
        except Exception as e:
            self.logger.error(f"❌ Error generating fallback description: {e}")
            return "Document processed successfully."


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
                    # Use explicit text mode to satisfy type checkers/newer fitz APIs
                    text_content += page.get_text("text") + "\n\n"  # type: ignore[attr-defined]
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
    
    async def process_directory_as_single_document(self, directory_path: Union[str, Path]) -> Optional[ProcessedDocument]:
        """Process an entire directory as a single document (for Obsidian vaults, doc collections)"""
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            self.logger.error(f"Directory not found: {directory_path}")
            return None
        
        if not directory_path.is_dir():
            self.logger.error(f"Path is not a directory: {directory_path}")
            return None
        
        self.logger.info(f"Processing directory as single document: {directory_path.name}")
        
        # Create processed document for the directory
        doc = ProcessedDocument(
            original_path=str(directory_path.absolute()),
            file_type="directory",
            file_size=0,
            status="processing"
        )
        
        try:
            # Collect all supported files in the directory
            file_paths = []
            total_size = 0
            combined_text = []
            
            for file_path in directory_path.rglob("*"):
                if file_path.is_file() and is_supported_file(file_path):
                    file_paths.append(str(file_path.relative_to(directory_path)))
                    total_size += file_path.stat().st_size
                    
                    # Extract text from each file
                    try:
                        file_text = await self._extract_text(file_path)
                        if file_text:
                            # Add file header to maintain context
                            relative_path = file_path.relative_to(directory_path)
                            combined_text.append(f"\n# File: {relative_path}\n\n{file_text}\n")
                            self.logger.debug(f"Added text from: {relative_path}")
                    except Exception as e:
                        self.logger.warning(f"Failed to extract text from {file_path}: {e}")
                        continue
            
            if not combined_text:
                doc.status = "failed"
                doc.error_message = "No extractable text found in directory"
                self.logger.error(f"❌ No extractable text found in: {directory_path.name}")
                return None
            
            # Combine all text content
            doc.text_content = "\n".join(combined_text)
            doc.file_size = total_size
            
            # Store metadata about the directory
            doc.metadata = {
                "is_directory": True,
                "sub_files": file_paths,
                "file_count": len(file_paths),
                "directory_name": directory_path.name,
                "processing_type": "directory_collection"
            }
            
            # Save processed text
            processed_dir = Path(self.config.processed_dir)
            processed_dir.mkdir(parents=True, exist_ok=True)
            processed_file = processed_dir / f"{doc.id}_{directory_path.name}.txt"
            processed_file.write_text(doc.text_content, encoding='utf-8')
            doc.processed_path = str(processed_file)
            doc.status = "completed"
            
            self.logger.info(f"✅ Successfully processed directory: {directory_path.name} ({len(file_paths)} files)")
            return doc
            
        except Exception as e:
            doc.status = "failed"
            doc.error_message = str(e)
            self.logger.error(f"❌ Error processing directory {directory_path.name}: {e}")
            return None


class DocumentKnowledgeGraph:
    """Handles individual document knowledge graphs - one graph per document"""
    
    def __init__(self, config: BookWormConfig, document_id: str, library_manager: Optional[LibraryManager] = None):
        self.config = config
        self.document_id = document_id
        self.logger = logging.getLogger(f"bookworm.doc_kg.{document_id[:8]}")
        self.rag: Any = None  # avoid type issues if LightRAG not available at type-check time
        self.library_manager = library_manager or LibraryManager(config)
        self.is_initialized = False
        
        # Create document-specific working directory in knowledge_graphs folder
        self.doc_working_dir = Path("./bookworm_workspace/knowledge_graphs") / document_id
        self.doc_working_dir.mkdir(parents=True, exist_ok=True)
        
        if not LightRAG:
            raise ImportError("LightRAG not available. Install lightrag-hku.")

    async def _robust_ollama_embed(self, texts, max_retries=3, delay=1):
        """Robust wrapper for ollama_embed with retry logic, from prototype."""
        from lightrag.llm.ollama import ollama_embed
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"Embedding attempt {attempt + 1}/{max_retries} with model {self.config.embedding_model}")
                return await ollama_embed(
                    texts, 
                    embed_model=self.config.embedding_model, 
                    host=self.config.embedding_host,
                    timeout=self.config.embedding_timeout,
                    options={"num_threads": 11}
                )
            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"Embedding failed after {max_retries} attempts: {e}")
                    raise e
                self.logger.warning(f"Embedding attempt {attempt + 1} failed: {e}, retrying in {delay}s...")
                await asyncio.sleep(delay)
                delay *= 2  # Exponential backoff
    
    async def initialize(self) -> None:
        """Initialize document-specific LightRAG system, mirroring prototype (Ollama only)."""
        if self.is_initialized:
            self.logger.info(f"Document KG {self.document_id[:8]} already initialized")
            return
            
        self.logger.info(f"Initializing document knowledge graph for {self.document_id[:8]}...")
        
        # Enforce Ollama-only usage as requested
        if str(getattr(self.config, "api_provider", "")).upper() != "OLLAMA":
            raise ValueError("BookWorm is configured to run only with Ollama. Please set API_PROVIDER=OLLAMA.")
        
        # Debug configuration
        self.logger.info("🔍 Configuration (Ollama-only):")
        self.logger.info(f"  Working Directory: {self.doc_working_dir}")
        self.logger.info(f"  LLM Model: {self.config.llm_model}")
        self.logger.info(f"  Embedding Model: {self.config.embedding_model}")
        self.logger.info(f"  LLM Host: {self.config.llm_host}")
        self.logger.info(f"  Embedding Host: {self.config.embedding_host}")

        # Enable verbose debug logging for LightRAG
        self.logger.info("🔍 Enabling verbose debug logging for knowledge graph processing...")
        if lightrag_set_verbose_debug:
            lightrag_set_verbose_debug(True)
        
        # Set LightRAG logger to DEBUG level
        lightrag_logger = logging.getLogger("lightrag")
        lightrag_logger.setLevel(logging.DEBUG)
        
        # Also enable debug for our own logger
        self.logger.setLevel(logging.DEBUG)

        from lightrag.utils import EmbeddingFunc
        from lightrag.llm.ollama import ollama_model_complete
        
        # Enhanced large-document settings from prototype - use ollama_model_complete directly like prototype
        self.rag = LightRAG(  # type: ignore
            working_dir=str(self.doc_working_dir),
            llm_model_func=ollama_model_complete,
            llm_model_name=self.config.llm_model,
            llm_model_kwargs={
                "host": self.config.llm_host,
                "options": {"num_ctx": 20192},
                "timeout": self.config.timeout,
            },
            embedding_func=EmbeddingFunc(
                embedding_dim=self.config.embedding_dim,
                max_token_size=getattr(self.config, "max_embed_tokens", 40000), # from prototype
                func=self._robust_ollama_embed,
            ),
            vector_storage="FaissVectorDBStorage",
            vector_db_storage_cls_kwargs={
                "cosine_better_than_threshold": 0.3
            },
        )
        
        # Initialize storage backends and pipeline
        self.logger.info("Initializing LightRAG storages...")
        await self.rag.initialize_storages()  # type: ignore[union-attr]
        if initialize_pipeline_status:
            await initialize_pipeline_status()
        
        self.is_initialized = True
        self.logger.info(f"✅ Document knowledge graph {self.document_id[:8]} initialized successfully")
    
    async def add_document_content(self, content: str, file_paths: Optional[List[str]] = None) -> None:
        """Add document content to this specific knowledge graph"""
        if not self.rag:
            raise RuntimeError("Knowledge graph not initialized. Call initialize() first.")
        
        if not content:
            self.logger.warning(f"Document {self.document_id[:8]} has no content, skipping insertion.")
            return
        
        try:
            self.logger.info(f"Inserting content into document KG {self.document_id[:8]} (size: {len(content):,} chars)...")
            self.logger.debug(f"📄 Content preview: {content[:200]}...")
            self.logger.debug("🔄 Starting LightRAG ainsert operation...")
            
            await self.rag.ainsert(content, file_paths=file_paths)
            
            self.logger.info(f"✅ Successfully inserted content into document KG {self.document_id[:8]}")
            self.logger.debug(f"📊 Knowledge graph creation completed for document {self.document_id[:8]}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to insert content into document KG {self.document_id[:8]}: {e}")
            self.logger.debug(f"🔍 Full error details: {type(e).__name__}: {str(e)}")
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
            self.logger.info(f"Querying document KG {self.document_id[:8]} in {mode} mode: '{query[:50]}...'")
            
            if stream:
                # Handle streaming response
                self.logger.debug("Awaiting streaming response...")
                result_stream = await self.rag.aquery(
                    query,
                    param=QueryParam(mode=mode, stream=stream, **kwargs),  # type: ignore
                )
                result = ""
                async for chunk in result_stream:
                    result += chunk
                self.logger.info("Streamed response received.")
                return result
            else:
                # Handle non-streaming response
                self.logger.debug("Awaiting non-streaming response...")
                result = await self.rag.aquery(
                    query,
                    param=QueryParam(mode=mode, stream=stream, **kwargs),  # type: ignore
                )
                self.logger.info("Non-streaming response received.")
                return result
                
        except Exception as e:
            self.logger.error(f"Query failed on document KG {self.document_id[:8]}: {e}")
            raise
    
    async def generate_description(self) -> str:
        """Generate a description of the document by querying its knowledge graph"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Query the knowledge graph to understand the document's content
            description_query = """What is this document about? Provide a concise summary (2-3 sentences) that includes:
1. The main topic or subject matter
2. The type of content (technical, academic, business, etc.)
3. Key themes or areas covered

Be specific and informative."""
            
            self.logger.info(f"Generating description for document {self.document_id[:8]} from knowledge graph...")
            
            # Use hybrid mode for best results combining local and global knowledge
            description = await self.query(description_query, mode="hybrid")
            
            if description and description.strip():
                # Clean up the description
                description = description.strip()
                
                # Remove any common prefixes the model might add
                prefixes_to_remove = [
                    "this document is about",
                    "this document discusses",
                    "the document covers",
                    "this text is about",
                    "the content focuses on",
                    "based on the content",
                    "according to the document"
                ]
                
                description_lower = description.lower()
                for prefix in prefixes_to_remove:
                    if description_lower.startswith(prefix):
                        description = description[len(prefix):].strip()
                        # Capitalize first letter after removing prefix
                        if description:
                            description = description[0].upper() + description[1:]
                        break
                
                self.logger.info(f"📝 Generated description from KG for document {self.document_id[:8]}: {description[:80]}...")
                return description
            else:
                # Fallback to a simple description
                fallback = f"Document processed and indexed in knowledge graph {self.document_id[:8]}."
                self.logger.warning(f"Empty description from KG, using fallback: {fallback}")
                return fallback
                
        except Exception as e:
            self.logger.error(f"Failed to generate description from KG for document {self.document_id[:8]}: {e}")
            # Return a basic fallback description
            return f"Document indexed in knowledge graph {self.document_id[:8]} with content analysis available for querying."
    
    async def generate_tags(self, max_tags: int = 8) -> list[str]:
        """Generate relevant tags for the document by querying its knowledge graph"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Query the knowledge graph to extract key topics and concepts for tags
            tags_query = f"""Based on the content in this document, extract {max_tags} relevant tags or keywords that best categorize and describe the document. 

Focus on:
1. Main topics and subject areas
2. Key concepts and terminology
3. Methodologies or approaches mentioned
4. Application domains or industries
5. Technical terms and processes

Provide ONLY a simple comma-separated list of tags (no explanations, no numbers, no bullets).
Examples: "machine learning, neural networks, data science, artificial intelligence, deep learning"

Make the tags specific and useful for categorization."""
            
            self.logger.info(f"Generating tags for document {self.document_id[:8]} from knowledge graph...")
            
            # Use hybrid mode for best results
            tags_response = await self.query(tags_query, mode="hybrid")
            
            if tags_response and tags_response.strip():
                # Clean up the response and extract tags
                tags_text = tags_response.strip()
                
                # Remove common prefixes that the AI might add
                prefixes_to_remove = [
                    "based on the document",
                    "the main tags are:",
                    "relevant tags:",
                    "keywords:",
                    "tags:",
                    "here are the tags:",
                    "the following tags"
                ]
                
                tags_lower = tags_text.lower()
                for prefix in prefixes_to_remove:
                    if tags_lower.startswith(prefix):
                        tags_text = tags_text[len(prefix):].strip()
                        if tags_text.startswith(":"):
                            tags_text = tags_text[1:].strip()
                        break
                
                # Split by comma and clean up each tag
                raw_tags = [tag.strip() for tag in tags_text.split(',')]
                
                # Clean and validate tags
                clean_tags = []
                for tag in raw_tags:
                    # Remove quotes, periods, and other unwanted characters
                    tag = tag.strip('"\'.,;:!?()[]{}').strip()
                    
                    # Skip empty tags, very short tags, or tags that are too long
                    if tag and 2 <= len(tag) <= 30:
                        # Convert to lowercase for consistency
                        tag = tag.lower()
                        
                        # Skip duplicates
                        if tag not in clean_tags:
                            clean_tags.append(tag)
                    
                    # Stop if we have enough tags
                    if len(clean_tags) >= max_tags:
                        break
                
                self.logger.info(f"🏷️  Generated {len(clean_tags)} tags from KG for document {self.document_id[:8]}: {clean_tags[:3]}...")
                return clean_tags
            else:
                # Fallback to basic tags based on document type
                fallback_tags = ["document", "processed"]
                self.logger.warning(f"Empty tags response from KG, using fallback: {fallback_tags}")
                return fallback_tags
                
        except Exception as e:
            self.logger.error(f"Failed to generate tags from KG for document {self.document_id[:8]}: {e}")
            # Return basic fallback tags
            return ["document", "processed"]


class KnowledgeGraph:
    """Manages multiple document knowledge graphs - one graph per document"""
    
    def __init__(self, config: BookWormConfig, library_manager: Optional[LibraryManager] = None):
        self.config = config
        self.logger = logging.getLogger("bookworm.knowledge_graph_manager")
        self.library_manager = library_manager or LibraryManager(config)
        self.document_graphs: Dict[str, DocumentKnowledgeGraph] = {}
        
        if not LightRAG:
            raise ImportError("LightRAG not available. Install lightrag-hku.")
    
    async def initialize(self) -> None:
        """Prepare knowledge graph manager (ensure workspace exists)."""
        # Turn on verbose LightRAG logging
        try:
            if lightrag_set_verbose_debug:
                lightrag_set_verbose_debug(True)
            if lightrag_setup_logger:
                lightrag_setup_logger("lightrag", "DEBUG", add_filter=False)
        except Exception:
            pass
        
        self.logger.info("Initializing KnowledgeGraph manager...")
        graphs_dir = Path("./bookworm_workspace/knowledge_graphs")
        graphs_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("KnowledgeGraph manager ready")
    
    async def finalize(self) -> None:
        """Finalize all document graphs and release resources."""
        self.logger.info("Finalizing KnowledgeGraph manager and all document graphs...")
        for doc_id, doc_kg in list(self.document_graphs.items()):
            try:
                if doc_kg.rag:
                    # Gracefully flush caches and close storages
                    try:
                        await doc_kg.rag.llm_response_cache.index_done_callback()
                    except Exception:
                        pass
                    await doc_kg.rag.finalize_storages()
                    self.logger.info(f"✅ Finalized KG for {doc_id[:8]}")
            except Exception as e:
                self.logger.warning(f"Failed to finalize KG {doc_id[:8]}: {e}")
    
    async def add_document(self, document: ProcessedDocument) -> Tuple[DocumentKnowledgeGraph, Optional[str]]:
        """Create a document graph and ingest its content."""
        return await self.create_document_graph(document)
    
    async def batch_add_documents(self, documents: List[ProcessedDocument]) -> List[Tuple[str, Optional[str]]]:
        """Batch add multiple documents to the knowledge graph system."""
        results: List[Tuple[str, Optional[str]]] = []
        for doc in documents:
            try:
                doc_kg, lib_id = await self.create_document_graph(doc)
                results.append((doc_kg.document_id, lib_id))
            except Exception as e:
                self.logger.warning(f"Failed to add document {doc.id[:8]}: {e}")
        return results
    
    async def create_document_graph(self, document: ProcessedDocument) -> Tuple[DocumentKnowledgeGraph, Optional[str]]:
        """Create a new knowledge graph for a specific document"""
        self.logger.info(f"Creating knowledge graph for document {document.id[:8]}...")
        
        # Create document-specific knowledge graph
        doc_kg = DocumentKnowledgeGraph(self.config, document.id, self.library_manager)
        await doc_kg.initialize()
        
        # Add document content to the graph (preserve original file path for metadata)
        try:
            original_path = document.original_path if document.original_path else None
            file_paths = [original_path] if original_path else None
        except Exception:
            file_paths = None
        
        self.logger.info(f"Adding document content for {document.id[:8]}...")
        await doc_kg.add_document_content(document.text_content, file_paths=file_paths)
        
        # Store reference to the graph
        self.document_graphs[document.id] = doc_kg
        
        # Add to library manager
        library_doc_id = None
        try:
            # Determine if this is a directory or file
            original_path = Path(document.original_path)
            is_directory = document.metadata.get("is_directory", False) or original_path.is_dir()
            
            if original_path.exists():
                # Check if document already exists in library
                search_name = original_path.name
                existing_docs = self.library_manager.find_documents(filename=search_name)
                
                if not existing_docs:
                    # Add new document to library
                    library_doc_id = self.library_manager.add_document(filepath=document.original_path)
                    self.logger.info(f"📚 Document {library_doc_id} added to library index")
                    
                    # If it's a directory, update the document record with directory metadata
                    if is_directory and document.metadata.get("sub_files"):
                        self.library_manager.update_document_metadata(
                            library_doc_id,
                            {
                                "is_directory": True,
                                "sub_files": document.metadata.get("sub_files", []),
                                "file_count": document.metadata.get("file_count", 0),
                                "processing_type": document.metadata.get("processing_type", "directory_collection")
                            }
                        )
                        self.logger.info(f"📁 Directory metadata saved for {search_name}")
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
                self.logger.warning(f"Original file/directory {document.original_path} not found, skipping library update")
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
    
    async def query(self, query: str, mode: str = "hybrid", **kwargs) -> str:
        """High-level query across available document graphs.
        If multiple graphs exist, aggregate results by document.
        """
        # Prefer loaded graphs; if none loaded, attempt to load all existing from disk
        if not self.document_graphs:
            graphs_dir = Path("./bookworm_workspace/knowledge_graphs")
            if graphs_dir.exists():
                for doc_dir in graphs_dir.iterdir():
                    if doc_dir.is_dir():
                        try:
                            await self.get_document_graph(doc_dir.name)
                        except Exception:
                            continue
        
        if not self.document_graphs:
            self.logger.warning("No document graphs available to query")
            return ""
        
        # If only one graph, return its response directly
        if len(self.document_graphs) == 1:
            only_doc = next(iter(self.document_graphs.values()))
            self.logger.info(f"Querying single available document graph: {only_doc.document_id[:8]}")
            return await only_doc.query(query, mode=mode, **kwargs)
        
        # Otherwise, query all and aggregate
        self.logger.info(f"Querying {len(self.document_graphs)} document graphs...")
        results = await self.query_all_documents(query, mode=mode, **kwargs)
        combined = []
        for doc_id, answer in results.items():
            if answer:
                combined.append(f"[Document {doc_id[:8]}]\n{answer}")
        return "\n\n".join(combined)
    
    async def query_document(self, document_id: str, query: str, mode: str = "hybrid", **kwargs) -> Optional[str]:
        """Query a specific document's knowledge graph"""
        doc_kg = await self.get_document_graph(document_id)
        if doc_kg:
            return await doc_kg.query(query, mode=mode, **kwargs)
        self.logger.warning(f"Could not find or load document graph for ID: {document_id}")
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
                        self.logger.debug(f"Querying loaded/found document graph: {document_id[:8]}")
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
            self.logger.warning(f"Document {document.id} has no text content, skipping mindmap generation.")
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
    
    async def batch_generate_mindmaps(self, documents: List[ProcessedDocument]) -> List[MindmapResult]:
        """Generate mindmaps for multiple documents."""
        results: List[MindmapResult] = []
        for doc in documents:
            try:
                if not doc.text_content:
                    self.logger.debug(f"Skipping mindmap for empty document: {doc.id[:8]}")
                    continue
                result = await self.generate_mindmap(doc)
                results.append(result)
            except Exception as e:
                self.logger.warning(f"Failed to generate mindmap for {doc.id[:8]}: {e}")
        return results
    
    async def _save_mindmap_files(self, result: MindmapResult, document: ProcessedDocument, document_library_id: Optional[str] = None) -> None:
        """Save mindmap files and update library"""
        doc_name = Path(document.original_path).stem
        
        # Add mindmap to library first to get the mindmap ID
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
            
            # Create temporary mindmap record to get ID
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
                mindmap_files={},  # Will update with actual files below
                metadata=mindmap_metadata
            )
            self.logger.info(f"📚 Mindmap {mindmap_id} added to library")
            
            # Now save files using the mindmap ID
            output_dir = Path("./bookworm_workspace/mindmaps") / mindmap_id
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
