"""
Document description generator using AI
"""
import logging
from typing import Optional

# Optional imports for LLM integration
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

from ..models import ProcessedDocument
from ..utils import BookWormConfig


class DocumentDescriptionGenerator:
    """Generate AI descriptions for documents"""
    
    def __init__(self, config: BookWormConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize Ollama client (same pattern as mindmap generator)
        self.ollama_client = None
        try:
            if AsyncOpenAI and self.config.api_provider == "OLLAMA":
                ollama_url = f"{self.config.llm_host}/v1"
                self.ollama_client = AsyncOpenAI(
                    api_key="ollama",  # Ollama doesn't require a real API key
                    base_url=ollama_url
                )
        except ImportError:
            self.logger.warning("OpenAI client not available for Ollama integration")
        
    async def generate_description(self, document: ProcessedDocument) -> Optional[str]:
        """Generate an AI description for a document"""
        try:
            # Prepare the text for description generation
            text_content = document.text_content
            
            # Truncate text if too long (keep first 2000 characters for context)
            if len(text_content) > 2000:
                text_content = text_content[:2000] + "..."
            
            # Determine document type for better prompting
            doc_type = "directory collection" if document.metadata.get("is_directory", False) else "document"
            file_info = ""
            if document.metadata.get("is_directory", False):
                file_count = document.metadata.get("file_count", 0)
                file_info = f" containing {file_count} files"
            
            # Create prompt for description generation
            prompt = f"""Analyze the following {doc_type}{file_info} and provide a concise, informative description (2-3 sentences) that captures:

1. The main topic or subject matter
2. The type of content (academic, technical, business, etc.)
3. Key themes or focus areas

Text to analyze:
{text_content}

Provide only the description, no additional formatting or labels."""

            # Generate description using Ollama client
            try:
                if self.ollama_client:
                    response = await self.ollama_client.chat.completions.create(
                        model=self.config.llm_model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=150,
                        temperature=0.3
                    )
                    
                    description = response.choices[0].message.content or ""
                    
                    if description:
                        # Clean up the description
                        description = description.strip()
                        # Remove any prefix like "Description:" if the model includes it
                        if description.lower().startswith("description:"):
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
