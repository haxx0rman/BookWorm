"""
Library state management for BookWorm
Tracks documents, mindmaps, and provides indexing functionality
"""
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum

from .utils import BookWormConfig


class DocumentStatus(Enum):
    """Document processing status"""
    PENDING = "pending"
    PROCESSING = "processing" 
    PROCESSED = "processed"
    FAILED = "failed"
    ARCHIVED = "archived"


class DocumentType(Enum):
    """Document types for categorization"""
    PDF = "pdf"
    TEXT = "text"
    MARKDOWN = "markdown"
    WORD = "word"
    POWERPOINT = "powerpoint"
    DIRECTORY = "directory"
    UNKNOWN = "unknown"


@dataclass
class DocumentRecord:
    """Record of a document in the library"""
    id: str
    filename: str
    filepath: str
    file_type: DocumentType
    file_size: int
    status: DocumentStatus
    created_at: datetime
    updated_at: datetime
    processed_at: Optional[datetime] = None
    
    # Processing results
    processed_file_path: Optional[str] = None
    knowledge_graph_id: Optional[str] = None
    mindmap_id: Optional[str] = None
    
    # Metadata
    title: Optional[str] = None
    author: Optional[str] = None
    description: Optional[str] = None  # AI-generated description of document content
    tags: Set[str] = field(default_factory=set)
    word_count: Optional[int] = None
    
    # Directory-specific fields
    is_directory: bool = False
    sub_files: List[str] = field(default_factory=list)  # List of file paths within directory
    file_count: Optional[int] = None  # Number of files in directory
    
    # Error tracking
    error_message: Optional[str] = None
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'filename': self.filename,
            'filepath': self.filepath,
            'file_type': self.file_type.value,
            'file_size': self.file_size,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'processed_at': self.processed_at.isoformat() if self.processed_at else None,
            'processed_file_path': self.processed_file_path,
            'knowledge_graph_id': self.knowledge_graph_id,
            'mindmap_id': self.mindmap_id,
            'title': self.title,
            'author': self.author,
            'description': self.description,
            'tags': list(self.tags),
            'word_count': self.word_count,
            'is_directory': self.is_directory,
            'sub_files': self.sub_files,
            'file_count': self.file_count,
            'error_message': self.error_message,
            'retry_count': self.retry_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentRecord':
        """Create from dictionary"""
        return cls(
            id=data['id'],
            filename=data['filename'],
            filepath=data['filepath'],
            file_type=DocumentType(data['file_type']),
            file_size=data['file_size'],
            status=DocumentStatus(data['status']),
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            processed_at=datetime.fromisoformat(data['processed_at']) if data.get('processed_at') else None,
            processed_file_path=data.get('processed_file_path'),
            knowledge_graph_id=data.get('knowledge_graph_id'),
            mindmap_id=data.get('mindmap_id'),
            title=data.get('title'),
            author=data.get('author'),
            description=data.get('description'),
            tags=set(data.get('tags', [])),
            word_count=data.get('word_count'),
            is_directory=data.get('is_directory', False),
            sub_files=data.get('sub_files', []),
            file_count=data.get('file_count'),
            error_message=data.get('error_message'),
            retry_count=data.get('retry_count', 0)
        )


@dataclass
class MindmapRecord:
    """Record of a mindmap in the library"""
    id: str
    document_id: str
    filename: str
    mermaid_file: str
    html_file: str
    markdown_file: str
    created_at: datetime
    updated_at: datetime
    
    # Generation metadata
    document_type: str
    token_usage: int
    processing_time: float
    generator_version: str
    
    # Content metadata
    topic_count: int
    subtopic_count: int
    detail_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'document_id': self.document_id,
            'filename': self.filename,
            'mermaid_file': self.mermaid_file,
            'html_file': self.html_file,
            'markdown_file': self.markdown_file,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'document_type': self.document_type,
            'token_usage': self.token_usage,
            'processing_time': self.processing_time,
            'generator_version': self.generator_version,
            'topic_count': self.topic_count,
            'subtopic_count': self.subtopic_count,
            'detail_count': self.detail_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MindmapRecord':
        """Create from dictionary"""
        return cls(
            id=data['id'],
            document_id=data['document_id'],
            filename=data['filename'],
            mermaid_file=data['mermaid_file'],
            html_file=data['html_file'],
            markdown_file=data['markdown_file'],
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            document_type=data['document_type'],
            token_usage=data['token_usage'],
            processing_time=data['processing_time'],
            generator_version=data['generator_version'],
            topic_count=data['topic_count'],
            subtopic_count=data['subtopic_count'],
            detail_count=data['detail_count']
        )


@dataclass
class LibraryStats:
    """Library statistics"""
    total_documents: int = 0
    processed_documents: int = 0
    pending_documents: int = 0
    failed_documents: int = 0
    total_mindmaps: int = 0
    total_size_bytes: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'total_documents': self.total_documents,
            'processed_documents': self.processed_documents,
            'pending_documents': self.pending_documents,
            'failed_documents': self.failed_documents,
            'total_mindmaps': self.total_mindmaps,
            'total_size_bytes': self.total_size_bytes,
            'last_updated': self.last_updated.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LibraryStats':
        """Create from dictionary"""
        return cls(
            total_documents=data.get('total_documents', 0),
            processed_documents=data.get('processed_documents', 0),
            pending_documents=data.get('pending_documents', 0),
            failed_documents=data.get('failed_documents', 0),
            total_mindmaps=data.get('total_mindmaps', 0),
            total_size_bytes=data.get('total_size_bytes', 0),
            last_updated=datetime.fromisoformat(data['last_updated']) if data.get('last_updated') else datetime.now()
        )


class LibraryManager:
    """Manages the BookWorm library state and indexing"""
    
    def __init__(self, config: BookWormConfig):
        self.config = config
        self.logger = logging.getLogger("bookworm.library")
        
        # Library file paths
        self.library_dir = Path(config.working_dir) / "library"
        self.library_dir.mkdir(exist_ok=True)
        
        self.documents_file = self.library_dir / "documents.json"
        self.mindmaps_file = self.library_dir / "mindmaps.json"
        self.stats_file = self.library_dir / "stats.json"
        self.index_file = self.library_dir / "index.json"
        
        # In-memory state
        self.documents: Dict[str, DocumentRecord] = {}
        self.mindmaps: Dict[str, MindmapRecord] = {}
        self.stats = LibraryStats()
        
        # Load existing state
        self._load_library_state()
        
        self.logger.info(f"📚 Library initialized with {len(self.documents)} documents and {len(self.mindmaps)} mindmaps")
    
    def _load_library_state(self):
        """Load library state from files"""
        try:
            # Load documents
            if self.documents_file.exists():
                with open(self.documents_file, 'r') as f:
                    documents_data = json.load(f)
                    self.documents = {
                        doc_id: DocumentRecord.from_dict(doc_data)
                        for doc_id, doc_data in documents_data.items()
                    }
            
            # Load mindmaps
            if self.mindmaps_file.exists():
                with open(self.mindmaps_file, 'r') as f:
                    mindmaps_data = json.load(f)
                    self.mindmaps = {
                        mindmap_id: MindmapRecord.from_dict(mindmap_data)
                        for mindmap_id, mindmap_data in mindmaps_data.items()
                    }
            
            # Load stats
            if self.stats_file.exists():
                with open(self.stats_file, 'r') as f:
                    stats_data = json.load(f)
                    self.stats = LibraryStats.from_dict(stats_data)
            
            # Update stats if needed
            self._update_stats()
            
        except Exception as e:
            self.logger.error(f"Failed to load library state: {e}")
    
    def reload_library_state(self):
        """Reload library state from files to get latest data"""
        self._load_library_state()
    
    def _save_library_state(self):
        """Save library state to files"""
        try:
            # Save documents
            documents_data = {
                doc_id: doc.to_dict()
                for doc_id, doc in self.documents.items()
            }
            with open(self.documents_file, 'w') as f:
                json.dump(documents_data, f, indent=2)
            
            # Save mindmaps
            mindmaps_data = {
                mindmap_id: mindmap.to_dict()
                for mindmap_id, mindmap in self.mindmaps.items()
            }
            with open(self.mindmaps_file, 'w') as f:
                json.dump(mindmaps_data, f, indent=2)
            
            # Save stats
            with open(self.stats_file, 'w') as f:
                json.dump(self.stats.to_dict(), f, indent=2)
            
            # Save search index
            self._save_search_index()
            
        except Exception as e:
            self.logger.error(f"Failed to save library state: {e}")
    
    def _update_stats(self):
        """Update library statistics"""
        self.stats.total_documents = len(self.documents)
        self.stats.processed_documents = len([d for d in self.documents.values() if d.status == DocumentStatus.PROCESSED])
        self.stats.pending_documents = len([d for d in self.documents.values() if d.status == DocumentStatus.PENDING])
        self.stats.failed_documents = len([d for d in self.documents.values() if d.status == DocumentStatus.FAILED])
        self.stats.total_mindmaps = len(self.mindmaps)
        self.stats.total_size_bytes = sum(d.file_size for d in self.documents.values())
        self.stats.last_updated = datetime.now()
    
    def _save_search_index(self):
        """Save search index for fast lookups"""
        index = {
            'documents_by_filename': {doc.filename: doc.id for doc in self.documents.values()},
            'documents_by_type': {},
            'documents_by_status': {},
            'mindmaps_by_document': {mindmap.document_id: mindmap.id for mindmap in self.mindmaps.values()},
            'tags': {}
        }
        
        # Index by document type
        for doc_type in DocumentType:
            index['documents_by_type'][doc_type.value] = [
                doc.id for doc in self.documents.values() if doc.file_type == doc_type
            ]
        
        # Index by status
        for status in DocumentStatus:
            index['documents_by_status'][status.value] = [
                doc.id for doc in self.documents.values() if doc.status == status
            ]
        
        # Index by tags
        all_tags = set()
        for doc in self.documents.values():
            all_tags.update(doc.tags)
        
        for tag in all_tags:
            index['tags'][tag] = [
                doc.id for doc in self.documents.values() if tag in doc.tags
            ]
        
        with open(self.index_file, 'w') as f:
            json.dump(index, f, indent=2)
    
    def add_document(self, filepath: str, filename: Optional[str] = None) -> str:
        """Add a new document to the library"""
        file_path = Path(filepath)
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {filepath}")
        
        # Generate document ID and record
        doc_id = str(uuid.uuid4())
        filename = filename or file_path.name
        
        # Determine file type - handle directories
        if file_path.is_dir():
            file_type = DocumentType.DIRECTORY
            file_size = sum(f.stat().st_size for f in file_path.rglob("*") if f.is_file())
            # Collect sub-files for directory processing
            sub_files = [
                str(f.relative_to(file_path)) for f in file_path.rglob("*") 
                if f.is_file() and f.suffix.lower() in {'.pdf', '.txt', '.md', '.markdown', '.docx', '.doc'}
            ]
            file_count = len(sub_files)
        else:
            file_type = self._get_file_type(file_path.suffix.lower())
            file_size = file_path.stat().st_size
            sub_files = []
            file_count = None
        
        # Create document record
        document = DocumentRecord(
            id=doc_id,
            filename=filename,
            filepath=str(file_path.absolute()),
            file_type=file_type,
            file_size=file_size,
            status=DocumentStatus.PENDING,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            is_directory=file_path.is_dir(),
            sub_files=sub_files,
            file_count=file_count
        )
        
        self.documents[doc_id] = document
        self._update_stats()
        self._save_library_state()
        
        doc_type = "directory" if file_path.is_dir() else "document"
        self.logger.info(f"📄 Added {doc_type} to library: {filename} (ID: {doc_id})")
        return doc_id
    
    def update_document_status(self, doc_id: str, status: DocumentStatus, 
                             processed_file_path: Optional[str] = None, 
                             knowledge_graph_id: Optional[str] = None,
                             error_message: Optional[str] = None):
        """Update document processing status"""
        if doc_id not in self.documents:
            raise ValueError(f"Document not found: {doc_id}")
        
        document = self.documents[doc_id]
        document.status = status
        document.updated_at = datetime.now()
        
        if status == DocumentStatus.PROCESSED:
            document.processed_at = datetime.now()
            if processed_file_path:
                document.processed_file_path = processed_file_path
            if knowledge_graph_id:
                document.knowledge_graph_id = knowledge_graph_id
        
        if status == DocumentStatus.FAILED:
            document.error_message = error_message
            document.retry_count += 1
        
        self._update_stats()
        self._save_library_state()
        
        self.logger.info(f"📄 Updated document {doc_id} status to {status.value}")
    
    def update_document_metadata(self, doc_id: str, metadata: Dict[str, Any]) -> None:
        """Update document metadata"""
        if doc_id not in self.documents:
            raise ValueError(f"Document not found: {doc_id}")
        
        document = self.documents[doc_id]
        
        # Update metadata fields that exist in DocumentRecord
        updateable_fields = {
            'knowledge_graph_path': 'knowledge_graph_id',  # Map path to id for storage
            'title': 'title',
            'author': 'author',
            'description': 'description',  # Add description field
            'word_count': 'word_count',
            'processed_file_path': 'processed_file_path',
            'knowledge_graph_id': 'knowledge_graph_id'
        }
        
        for key, value in metadata.items():
            if key in updateable_fields:
                field_name = updateable_fields[key]
                if hasattr(document, field_name):
                    setattr(document, field_name, value)
            elif key == 'tags' and isinstance(value, (list, set)):
                # Handle tags specially
                document.tags.update(value)
        
        document.updated_at = datetime.now()
        
        self._update_stats()
        self._save_library_state()
        
        self.logger.info(f"📄 Updated document {doc_id} metadata")
    
    def add_mindmap(self, document_id: str, mindmap_files: Dict[str, str], 
                   metadata: Dict[str, Any]) -> str:
        """Add a mindmap record to the library"""
        if document_id not in self.documents:
            raise ValueError(f"Document not found: {document_id}")
        
        # Generate mindmap ID
        mindmap_id = str(uuid.uuid4())
        
        # Create mindmap record
        mindmap = MindmapRecord(
            id=mindmap_id,
            document_id=document_id,
            filename=f"mindmap_{self.documents[document_id].filename}",
            mermaid_file=mindmap_files.get('mermaid', ''),
            html_file=mindmap_files.get('html', ''),
            markdown_file=mindmap_files.get('markdown', ''),
            created_at=datetime.now(),
            updated_at=datetime.now(),
            document_type=metadata.get('document_type', 'unknown'),
            token_usage=metadata.get('token_usage', 0),
            processing_time=metadata.get('processing_time', 0.0),
            generator_version=metadata.get('generator_version', '1.0'),
            topic_count=metadata.get('topic_count', 0),
            subtopic_count=metadata.get('subtopic_count', 0),
            detail_count=metadata.get('detail_count', 0)
        )
        
        self.mindmaps[mindmap_id] = mindmap
        
        # Update document record
        self.documents[document_id].mindmap_id = mindmap_id
        
        self._update_stats()
        self._save_library_state()
        
        self.logger.info(f"🗺️ Added mindmap to library: {mindmap.filename} (ID: {mindmap_id})")
        return mindmap_id
    
    def get_document(self, doc_id: str) -> Optional[DocumentRecord]:
        """Get document by ID"""
        return self.documents.get(doc_id)
    
    def get_mindmap(self, mindmap_id: str) -> Optional[MindmapRecord]:
        """Get mindmap by ID"""
        return self.mindmaps.get(mindmap_id)
    
    def get_document_by_filename(self, filename: str) -> Optional[DocumentRecord]:
        """Get document by filename"""
        for document in self.documents.values():
            if document.filename == filename:
                return document
        return None
    
    def find_documents(self, filename: Optional[str] = None, status: Optional[DocumentStatus] = None, 
                      file_type: Optional[DocumentType] = None, tags: Optional[List[str]] = None) -> List[DocumentRecord]:
        """Find documents by criteria"""
        results = list(self.documents.values())
        
        if filename:
            results = [doc for doc in results if filename.lower() in doc.filename.lower()]
        
        if status:
            results = [doc for doc in results if doc.status == status]
        
        if file_type:
            results = [doc for doc in results if doc.file_type == file_type]
        
        if tags:
            results = [doc for doc in results if any(tag in doc.tags for tag in tags)]
        
        return results
    
    def get_library_stats(self) -> LibraryStats:
        """Get current library statistics"""
        self._update_stats()
        return self.stats
    
    def scan_directories(self) -> List[str]:
        """Scan document directories for new files"""
        new_documents = []
        
        # Get existing files from library
        existing_paths = {doc.filepath for doc in self.documents.values()}
        
        # Scan document directory
        doc_dir = Path(self.config.document_dir)
        if doc_dir.exists():
            for file_path in doc_dir.rglob('*'):
                if file_path.is_file() and str(file_path.absolute()) not in existing_paths:
                    # Check if it's a supported file type
                    if self._is_supported_file(file_path):
                        doc_id = self.add_document(str(file_path))
                        new_documents.append(doc_id)
        
        if new_documents:
            self.logger.info(f"📁 Discovered {len(new_documents)} new documents during directory scan")
        
        return new_documents
    
    def _get_file_type(self, extension: str) -> DocumentType:
        """Determine file type from extension"""
        type_map = {
            '.pdf': DocumentType.PDF,
            '.txt': DocumentType.TEXT,
            '.md': DocumentType.MARKDOWN,
            '.markdown': DocumentType.MARKDOWN,
            '.docx': DocumentType.WORD,
            '.doc': DocumentType.WORD,
            '.pptx': DocumentType.POWERPOINT,
            '.ppt': DocumentType.POWERPOINT,
            'directory': DocumentType.DIRECTORY  # Special case for directories
        }
        return type_map.get(extension, DocumentType.UNKNOWN)
    
    def _is_supported_file(self, file_path: Path) -> bool:
        """Check if file type is supported"""
        supported_extensions = {'.pdf', '.txt', '.md', '.markdown', '.docx', '.doc', '.html', '.htm'}
        return file_path.suffix.lower() in supported_extensions
    
    def export_library_report(self, output_file: Optional[str] = None) -> str:
        """Export a comprehensive library report"""
        if not output_file:
            output_file = str(self.library_dir / f"library_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'stats': self.stats.to_dict(),
            'documents': [doc.to_dict() for doc in self.documents.values()],
            'mindmaps': [mindmap.to_dict() for mindmap in self.mindmaps.values()],
            'summary': {
                'total_documents': len(self.documents),
                'by_status': {
                    status.value: len([d for d in self.documents.values() if d.status == status])
                    for status in DocumentStatus
                },
                'by_type': {
                    file_type.value: len([d for d in self.documents.values() if d.file_type == file_type])
                    for file_type in DocumentType
                },
                'mindmap_coverage': f"{(len(self.mindmaps) / max(1, len(self.documents))) * 100:.1f}%"
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"📊 Exported library report to: {output_file}")
        return str(output_file)
