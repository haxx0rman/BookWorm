"""
BookWorm - Advanced Document/Knowledge Ingestion System
Integrating LightRAG with Mindmap Generation for comprehensive document processing
"""

__version__ = "0.1.0"
__author__ = "BookWorm Team"
__description__ = "Advanced document/knowledge ingestion system with LightRAG and mindmap generation"

from .core import DocumentProcessor, KnowledgeGraph, MindmapGenerator
from .utils import setup_logging, load_config

__all__ = [
    "DocumentProcessor",
    "KnowledgeGraph", 
    "MindmapGenerator",
    "setup_logging",
    "load_config",
]
