"""
Mindmap generator using advanced mindmap generation
"""
import logging
from pathlib import Path
from typing import List, Optional

from ..models import ProcessedDocument, MindmapResult
from ..library import LibraryManager
from ..mindmap_generator import AdvancedMindmapGenerator
from ..utils import BookWormConfig


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
        
        # Try to add/update in library if possible, but always save files
        mindmap_id = None
        doc_id = None
        
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
                    # Document not in library - create a simple mindmap ID
                    mindmap_id = f"mindmap_{document.id[:8]}"
                    self.logger.warning(f"Document not in library, using standalone mindmap ID: {mindmap_id}")
            
            # Create mindmap record in library if document exists
            if doc_id:
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
            
        except Exception as e:
            # If library operations fail, still save files with fallback ID
            if not mindmap_id:
                mindmap_id = f"mindmap_{document.id[:8]}"
            self.logger.warning(f"Library operations failed, using fallback mindmap ID: {mindmap_id}")
            self.logger.warning(f"Library error: {e}")
        
        # Ensure we have a mindmap_id
        if not mindmap_id:
            mindmap_id = f"mindmap_{document.id[:8]}"
        
        # Always save files regardless of library status
        try:
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
            
            self.logger.info(f"✅ Mindmap files saved to: {output_dir}")
            
            # Update library with file paths if document exists
            if doc_id and mindmap_files:
                try:
                    self.library_manager.update_document_metadata(
                        doc_id, 
                        {"mindmap_id": mindmap_id}
                    )
                    self.logger.info(f"📚 Document {doc_id} updated with mindmap ID: {mindmap_id}")
                except Exception as e:
                    self.logger.warning(f"Failed to update library metadata: {e}")
            
        except Exception as e:
            self.logger.error(f"Failed to save mindmap files: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
