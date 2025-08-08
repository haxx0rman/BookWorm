"""
Configuration management and utility functions for BookWorm
"""
import os
import logging
from pathlib import Path
from typing import Dict, Optional, Union
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(dotenv_path=None):
        """Fallback if python-dotenv is not available"""
        pass


@dataclass
class BookWormConfig:
    """Configuration class for BookWorm system following user's lightrag methodology"""
    
    # API Keys
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    deepseek_api_key: str = ""
    gemini_api_key: str = ""
    api_provider: str = "OPENAI"
    
    # LightRAG Settings (following user's lightrag_ex.py and lightrag_manager.py exactly)
    llm_model: str = "qwen2.5-coder:32b"  # User's model from lightrag_manager.py
    embedding_model: str = "bge-m3:latest"  # User's embedding model
    llm_host: str = "http://100.95.157.120:11434"  # User's specific host
    embedding_host: str = "http://100.95.157.120:11434"  # User's specific host
    embedding_dim: int = 1024  # User's dimension (not 1536)
    max_embed_tokens: int = 8192  # User's max tokens
    timeout: int = 3000  # User's timeout (3000 seconds, not 300)
    embedding_timeout: int = 6000  # User's embedding timeout (6000 seconds, not 600)
    
    # Directory Settings (following user's lightrag_ex.py structure)
    working_dir: str = "./lightrag_workspace"  # User's exact working directory
    document_dir: str = "./lightrag_workspace/docs"
    processed_dir: str = "./lightrag_workspace/processed_docs"  # Following user's DOCUMENT_ARCHIVE_DIR
    output_dir: str = "./lightrag_workspace/output"
    
    # PDF Processing (following user's preference for mineru)
    pdf_processor: str = "mineru"  # Primary: mineru, fallback: pymupdf, pdfplumber
    skip_pdf_conversion: bool = False
    
    # Logging
    log_level: str = "INFO"
    log_dir: str = "./logs"
    log_max_bytes: int = 10485760
    log_backup_count: int = 5
    
    # Performance
    max_concurrent_processes: int = 4
    chunk_size: int = 8192
    max_file_size_mb: int = 100


def load_config(env_file: Optional[str] = None) -> BookWormConfig:
    """Load configuration from environment variables"""
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()
    
    config = BookWormConfig()
    
    # Load API keys
    config.openai_api_key = os.getenv("OPENAI_API_KEY", "")
    config.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
    config.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "")
    config.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
    config.api_provider = os.getenv("API_PROVIDER", "OPENAI")
    
    # Load LightRAG settings
    config.llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    config.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    config.llm_host = os.getenv("LLM_HOST", "http://localhost:11434")
    config.embedding_host = os.getenv("EMBEDDING_HOST", "http://localhost:11434")
    config.embedding_dim = int(os.getenv("EMBEDDING_DIM", "1536"))
    config.max_embed_tokens = int(os.getenv("MAX_EMBED_TOKENS", "8192"))
    config.timeout = int(os.getenv("TIMEOUT", "300"))
    config.embedding_timeout = int(os.getenv("EMBEDDING_TIMEOUT", "600"))
    
    # Load directory settings
    config.working_dir = os.getenv("WORKING_DIR", "./bookworm_workspace")
    config.document_dir = os.getenv("DOCUMENT_DIR", "./bookworm_workspace/docs")
    config.processed_dir = os.getenv("PROCESSED_DIR", "./bookworm_workspace/processed")
    config.output_dir = os.getenv("OUTPUT_DIR", "./bookworm_workspace/output")
    
    # Load PDF processing settings
    config.pdf_processor = os.getenv("PDF_PROCESSOR", "pymupdf")
    config.skip_pdf_conversion = os.getenv("SKIP_PDF_CONVERSION", "false").lower() == "true"
    
    # Load logging settings
    config.log_level = os.getenv("LOG_LEVEL", "INFO")
    config.log_dir = os.getenv("LOG_DIR", "./logs")
    config.log_max_bytes = int(os.getenv("LOG_MAX_BYTES", "10485760"))
    config.log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", "5"))
    
    # Load performance settings
    config.max_concurrent_processes = int(os.getenv("MAX_CONCURRENT_PROCESSES", "4"))
    config.chunk_size = int(os.getenv("CHUNK_SIZE", "8192"))
    config.max_file_size_mb = int(os.getenv("MAX_FILE_SIZE_MB", "100"))
    
    return config


def setup_logging(config: BookWormConfig) -> logging.Logger:
    """Set up logging configuration"""
    from logging.handlers import RotatingFileHandler
    
    # Create log directory if it doesn't exist
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    
    # Configure logger
    logger = logging.getLogger("bookworm")
    logger.setLevel(getattr(logging, config.log_level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, config.log_level.upper()))
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = Path(config.log_dir) / "bookworm.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=config.log_max_bytes,
        backupCount=config.log_backup_count
    )
    file_handler.setLevel(getattr(logging, config.log_level.upper()))
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger


def ensure_directories(config: BookWormConfig) -> None:
    """Ensure all required directories exist"""
    directories = [
        config.working_dir,
        config.document_dir,
        config.processed_dir,
        config.output_dir,
        config.log_dir,
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def get_supported_extensions() -> Dict[str, list]:
    """Get supported file extensions by category"""
    return {
        "pdf": [".pdf"],
        "text": [".txt", ".md", ".markdown"],
        "document": [".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls"],
        "code": [".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".cs"],
        "data": [".json", ".yaml", ".yml", ".xml", ".csv"],
    }


def is_supported_file(file_path: Union[str, Path]) -> bool:
    """Check if file type is supported"""
    if isinstance(file_path, str):
        file_path = Path(file_path)
    extension = file_path.suffix.lower()
    
    supported_extensions = get_supported_extensions()
    all_extensions = []
    for extensions_list in supported_extensions.values():
        all_extensions.extend(extensions_list)
    
    return extension in all_extensions


def get_file_category(file_path: Union[str, Path]) -> Optional[str]:
    """Get the category of a file based on its extension"""
    if isinstance(file_path, str):
        file_path = Path(file_path)
    extension = file_path.suffix.lower()
    
    supported_extensions = get_supported_extensions()
    for category, extensions in supported_extensions.items():
        if extension in extensions:
            return category
    
    return None
