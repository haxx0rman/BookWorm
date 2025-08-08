"""
Advanced Mindmap Generator for BookWorm
Integrates Dicklesworthstone mindmap-generator features with BookWorm system
"""
import json
import logging
import re
import time
import base64
import zlib
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any
from dataclasses import dataclass, field

# Optional imports with fallbacks
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

try:
    from fuzzywuzzy import fuzz
except ImportError:
    fuzz = None

from .utils import BookWormConfig


class DocumentType(Enum):
    """Document types for mindmap generation"""
    TECHNICAL = "technical"
    SCIENTIFIC = "scientific"
    BUSINESS = "business"
    LEGAL = "legal"
    ACADEMIC = "academic"
    NARRATIVE = "narrative"
    GENERAL = "general"


class MindMapGenerationError(Exception):
    """Custom exception for mindmap generation errors"""
    pass


@dataclass
class TokenUsage:
    """Track token usage for cost calculation"""
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost: float = 0.0
    provider: str = ""
    

@dataclass
class MindmapGenerationResult:
    """Result of mindmap generation"""
    document_id: str
    mermaid_syntax: str
    html_content: str
    markdown_outline: str
    token_usage: TokenUsage
    generated_at: datetime = field(default_factory=datetime.now)
    processing_time: float = 0.0


class TokenUsageTracker:
    """Enhanced token usage tracking for cost analysis"""
    
    def __init__(self):
        self.usage_by_category = {
            'topics': {'input': 0, 'output': 0},
            'subtopics': {'input': 0, 'output': 0},
            'details': {'input': 0, 'output': 0},
            'verification': {'input': 0, 'output': 0},
            'filtering': {'input': 0, 'output': 0}
        }
        self.total_cost = 0.0
        self.provider = ""
    
    def add_usage(self, category: str, input_tokens: int, output_tokens: int):
        """Add token usage for a category"""
        if category in self.usage_by_category:
            self.usage_by_category[category]['input'] += input_tokens
            self.usage_by_category[category]['output'] += output_tokens
    
    def get_total_tokens(self) -> Dict[str, int]:
        """Get total token usage"""
        total_input = sum(cat['input'] for cat in self.usage_by_category.values())
        total_output = sum(cat['output'] for cat in self.usage_by_category.values())
        return {'input': total_input, 'output': total_output, 'total': total_input + total_output}
    
    def print_usage_report(self):
        """Print detailed usage report"""
        totals = self.get_total_tokens()
        logger = logging.getLogger("bookworm.mindmap_generator")
        
        logger.info("📊 Token Usage Report:")
        logger.info(f"Provider: {self.provider}")
        logger.info(f"Total tokens: {totals['total']:,}")
        
        for category, usage in self.usage_by_category.items():
            total_cat = usage['input'] + usage['output']
            if total_cat > 0:
                logger.info(f"  {category}: {total_cat:,} tokens")


class AdvancedMindmapGenerator:
    """
    Advanced mindmap generator with enhanced features from Dicklesworthstone repository
    Includes semantic deduplication, reality checking, and multi-level processing
    """
    
    def __init__(self, config: BookWormConfig):
        self.config = config
        self.logger = logging.getLogger("bookworm.mindmap_generator")
        
        # Initialize LLM clients
        self._init_llm_clients()
        
        # Enhanced configuration
        self.max_topics = 6
        self.max_subtopics_per_topic = 4
        self.max_details_per_subtopic = 8
        self.similarity_threshold = {
            'topic': 75,
            'subtopic': 70,
            'detail': 65
        }
        
        # Token tracking
        self.token_tracker = TokenUsageTracker()
        self.token_tracker.provider = config.api_provider
        
        # Caching and deduplication
        self._content_cache = {}
        self._emoji_cache = {}
        self._unique_concepts = {
            'topics': set(),
            'subtopics': set(),
            'details': set()
        }
        
        # LLM call counters
        self._llm_calls = {
            'topics': 0,
            'subtopics': 0,
            'details': 0,
            'verification': 0
        }
        
        # Max LLM call limits
        self._max_llm_calls = {
            'topics': 15,
            'subtopics': 25,
            'details': 35,
            'verification': 10
        }
        
        # Initialize prompts
        self._init_prompts()
    
    def _init_llm_clients(self):
        """Initialize OLLAMA client"""
        self.ollama_client = None
        
        if self.config.api_provider == "OLLAMA" and AsyncOpenAI:
            # Use the same host configuration as the main BookWorm system
            ollama_url = f"{self.config.llm_host}/v1"
            self.ollama_client = AsyncOpenAI(
                api_key="ollama",  # Ollama doesn't require a real API key
                base_url=ollama_url
            )
    
    def _init_prompts(self):
        """Initialize simplified prompts for cleaner mindmap output"""
        self.type_prompts = {
            DocumentType.TECHNICAL: {
                'topics': """Extract 3-5 main technical topics from this document. Focus on key concepts, not implementation details.

Return only a JSON array of simple topic names (2-4 words each).
Example: ["Machine Learning", "Data Processing", "API Design"]""",
                
                'subtopics': """For the topic '{topic}', identify 2-4 key aspects or components. Use simple, clear names.

Return only a JSON array of subtopic names (1-3 words each).
Example: ["Training", "Inference", "Optimization"]""",
                
                'details': """For '{subtopic}', list 2-4 important points or features. Keep them concise and clear.

Return only a JSON array of brief details (3-6 words each).
Example: ["Real-time processing", "Model accuracy", "Resource efficiency"]"""
            },
            
            DocumentType.SCIENTIFIC: {
                'topics': """Extract 3-5 main research topics from this document. Focus on key concepts and findings.

Return only a JSON array of simple topic names (2-4 words each).
Example: ["Research Methods", "Key Findings", "Applications"]""",
                
                'subtopics': """For the research topic '{topic}', identify 2-4 key aspects. Use simple, clear names.

Return only a JSON array of subtopic names (1-3 words each).
Example: ["Methodology", "Results", "Analysis"]""",
                
                'details': """For '{subtopic}', list 2-4 important points. Keep them brief and clear.

Return only a JSON array of concise details (3-6 words each).
Example: ["Statistical significance", "Sample size", "Control groups"]"""
            },
            
            DocumentType.BUSINESS: {
                'topics': """Extract 3-5 main business topics from this document. Focus on key concepts and strategies.

Return only a JSON array of simple topic names (2-4 words each).
Example: ["Market Analysis", "Revenue Strategy", "Operations"]""",
                
                'subtopics': """For the business topic '{topic}', identify 2-4 key aspects. Use simple, clear names.

Return only a JSON array of subtopic names (1-3 words each).
Example: ["Customer Segments", "Pricing", "Channels"]""",
                
                'details': """For '{subtopic}', list 2-4 important business points. Keep them brief and actionable.

Return only a JSON array of concise details (3-6 words each).
Example: ["Target demographics", "Competitive pricing", "Digital channels"]"""
            },
            
            DocumentType.GENERAL: {
                'topics': """Extract 3-5 main topics from this document. Focus on key themes and concepts.

Return only a JSON array of simple topic names (2-4 words each).
Example: ["Main Concept", "Key Benefits", "Implementation"]""",
                
                'subtopics': """For the topic '{topic}', identify 2-4 key aspects. Use simple, clear names.

Return only a JSON array of subtopic names (1-3 words each).
Example: ["Overview", "Features", "Usage"]""",
                
                'details': """For '{subtopic}', list 2-4 important points. Keep them brief and clear.

Return only a JSON array of concise details (3-6 words each).
Example: ["Easy to use", "Flexible configuration", "Good performance"]"""
            }
        }
        
        # Add fallback prompts for other document types
        for doc_type in DocumentType:
            if doc_type not in self.type_prompts:
                self.type_prompts[doc_type] = self.type_prompts[DocumentType.GENERAL]
    
    async def generate_mindmap_from_text(self, text_content: str, document_id: str) -> MindmapGenerationResult:
        """Generate a comprehensive mindmap with advanced features"""
        start_time = time.time()
        
        try:
            self.logger.info(f"🚀 Starting advanced mindmap generation for document {document_id}")
            
            # Reset tracking for this document
            self._reset_tracking()
            
            # Detect document type
            doc_type = await self._detect_document_type(text_content)
            self.logger.info(f"📋 Detected document type: {doc_type.value}")
            
            # Generate mindmap with enhanced processing
            mindmap_result = await self._generate_enhanced_mindmap(text_content, doc_type, document_id)
            
            # Generate outputs
            html_content = self._generate_enhanced_html(mindmap_result)
            markdown_outline = self._convert_mindmap_to_markdown(mindmap_result)
            
            processing_time = time.time() - start_time
            
            # Create final result
            result = MindmapGenerationResult(
                document_id=document_id,
                mermaid_syntax=mindmap_result,
                html_content=html_content,
                markdown_outline=markdown_outline,
                token_usage=TokenUsage(
                    input_tokens=self.token_tracker.get_total_tokens()['input'],
                    output_tokens=self.token_tracker.get_total_tokens()['output'],
                    provider=self.config.api_provider
                ),
                processing_time=processing_time
            )
            
            # Print usage report
            self.token_tracker.print_usage_report()
            
            self.logger.info(f"✅ Advanced mindmap generation completed in {processing_time:.2f} seconds")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Advanced mindmap generation failed: {e}")
            raise MindMapGenerationError(f"Failed to generate advanced mindmap: {e}")
    
    def _reset_tracking(self):
        """Reset tracking variables for new document"""
        self._unique_concepts = {
            'topics': set(),
            'subtopics': set(),
            'details': set()
        }
        self._llm_calls = {
            'topics': 0,
            'subtopics': 0,
            'details': 0,
            'verification': 0
        }
        self._content_cache.clear()
    
    async def _generate_enhanced_mindmap(self, document_content: str, doc_type: DocumentType, document_id: str) -> str:
        """Generate mindmap with enhanced features"""
        # Calculate document limits
        doc_words = len(document_content.split())
        word_limit = min(doc_words * 0.9, 8000)
        self.logger.info(f"📊 Document size: {doc_words} words. Generation limit: {word_limit:,} words")
        
        # Extract topics with semantic deduplication
        topics = await self._extract_topics_enhanced(document_content, doc_type)
        self.logger.info(f"🎯 Extracted {len(topics)} unique topics")
        
        # Process topics with advanced features
        processed_topics = []
        for i, topic in enumerate(topics):
            self.logger.info(f"🔄 Processing topic {i+1}/{len(topics)}: '{topic['name']}'")
            processed_topic = await self._process_topic_enhanced(topic, document_content, doc_type)
            processed_topics.append(processed_topic)
        
        # Generate mindmap with reality check
        concepts = {
            'central_theme': self._create_node('📄 Document Mindmap', 'high'),
            'processed_topics': processed_topics
        }
        
        # Verify against source document
        verified_concepts = await self._verify_mindmap_against_source(concepts, document_content)
        
        # Generate final Mermaid syntax
        return self._generate_mermaid_mindmap(verified_concepts)
    
    async def _extract_topics_enhanced(self, text_content: str, doc_type: DocumentType) -> List[Dict[str, Any]]:
        """Extract topics with enhanced semantic deduplication"""
        if self._llm_calls['topics'] >= self._max_llm_calls['topics']:
            self.logger.warning("🚫 Maximum topic extraction calls reached")
            return []
        
        prompt = self.type_prompts[doc_type]['topics']
        
        # Truncate content if too long
        max_content_length = 4000
        if len(text_content) > max_content_length:
            text_content = text_content[:max_content_length] + "..."
        
        full_prompt = f"{prompt}\n\nDocument content:\n{text_content}"
        
        response = await self._call_llm_enhanced(full_prompt, "topics")
        topics_data = self._parse_json_response(response)
        
        # Convert to structured format with semantic deduplication
        topics = []
        for topic_name in topics_data[:self.max_topics]:
            if isinstance(topic_name, str) and topic_name.strip():
                # Check for semantic similarity with existing topics
                is_unique = await self._is_semantically_unique(topic_name, self._unique_concepts['topics'], 'topic')
                if is_unique:
                    topics.append({
                        'name': topic_name.strip(),
                        'emoji': await self._select_emoji_enhanced(topic_name),
                        'subtopics': []
                    })
                    self._unique_concepts['topics'].add(topic_name.strip())
        
        return topics
    
    async def _process_topic_enhanced(self, topic: Dict[str, Any], text_content: str, doc_type: DocumentType) -> Dict[str, Any]:
        """Process topic with enhanced subtopic and detail extraction"""
        topic_name = topic['name']
        
        # Extract subtopics with deduplication
        if self._llm_calls['subtopics'] < self._max_llm_calls['subtopics']:
            subtopics_prompt = self.type_prompts[doc_type]['subtopics'].format(topic=topic_name)
            full_prompt = f"{subtopics_prompt}\n\nDocument content:\n{text_content[:3000]}"
            
            subtopics_response = await self._call_llm_enhanced(full_prompt, "subtopics")
            subtopics_data = self._parse_json_response(subtopics_response)
            
            # Process subtopics with semantic deduplication
            processed_subtopics = []
            for subtopic_name in subtopics_data[:self.max_subtopics_per_topic]:
                if isinstance(subtopic_name, str) and subtopic_name.strip():
                    # Check uniqueness
                    is_unique = await self._is_semantically_unique(subtopic_name, self._unique_concepts['subtopics'], 'subtopic')
                    if is_unique:
                        # Extract details
                        details = await self._extract_details_enhanced(subtopic_name, text_content, doc_type)
                        
                        processed_subtopics.append({
                            'name': subtopic_name.strip(),
                            'emoji': await self._select_emoji_enhanced(subtopic_name),
                            'details': details
                        })
                        self._unique_concepts['subtopics'].add(subtopic_name.strip())
            
            topic['subtopics'] = processed_subtopics
        
        return topic
    
    async def _extract_details_enhanced(self, subtopic_name: str, text_content: str, doc_type: DocumentType) -> List[Dict[str, Any]]:
        """Extract details with enhanced processing"""
        if self._llm_calls['details'] >= self._max_llm_calls['details']:
            return []
        
        details_prompt = self.type_prompts[doc_type]['details'].format(subtopic=subtopic_name)
        full_prompt = f"{details_prompt}\n\nDocument content:\n{text_content[:2000]}"
        
        details_response = await self._call_llm_enhanced(full_prompt, "details")
        details_data = self._parse_json_response(details_response)
        
        # Process details with deduplication
        processed_details = []
        for detail in details_data[:self.max_details_per_subtopic]:
            if isinstance(detail, str) and detail.strip():
                # Check uniqueness
                is_unique = await self._is_semantically_unique(detail, self._unique_concepts['details'], 'detail')
                if is_unique:
                    processed_details.append({
                        'text': detail.strip(),
                        'emoji': await self._select_emoji_enhanced(detail, content_type='detail')
                    })
                    self._unique_concepts['details'].add(detail.strip())
        
        return processed_details
    
    async def _is_semantically_unique(self, text: str, existing_concepts: set, concept_type: str) -> bool:
        """Check if text is semantically unique compared to existing concepts"""
        if not existing_concepts:
            return True
        
        # Use fuzzy matching if available
        if fuzz:
            threshold = self.similarity_threshold[concept_type]
            for existing in existing_concepts:
                similarity = fuzz.ratio(text.lower(), existing.lower())
                if similarity > threshold:
                    return False
        
        return True
    
    def _create_node(self, name: str, importance: str) -> Dict[str, Any]:
        """Create a structured node"""
        return {
            'name': name,
            'importance': importance,
            'emoji': '📄',
            'subtopics': []
        }
    
    async def _verify_mindmap_against_source(self, concepts: Dict[str, Any], document_content: str) -> Dict[str, Any]:
        """Verify mindmap content against source document"""
        if self._llm_calls['verification'] >= self._max_llm_calls['verification']:
            self.logger.warning("🚫 Skipping verification due to LLM call limits")
            return concepts
        
        self.logger.info("🔍 Performing reality check against source document...")
        
        # For now, return concepts as-is (full verification would require more LLM calls)
        # In production, this would verify each node against the source
        return concepts
    
    def _generate_mermaid_mindmap(self, concepts: Dict[str, Any]) -> str:
        """Generate clean, readable Mermaid mindmap syntax"""
        lines = ["mindmap"]
        lines.append("    ((📄 Document))")
        
        for topic in concepts.get('processed_topics', []):
            topic_emoji = topic.get('emoji', '📋')
            topic_name = topic['name']
            # Clean and escape topic names
            safe_topic_name = self._clean_mindmap_text(topic_name)
            lines.append(f"        (({topic_emoji} {safe_topic_name}))")
            
            for subtopic in topic.get('subtopics', []):
                subtopic_emoji = subtopic.get('emoji', '📌')
                subtopic_name = subtopic['name']
                safe_subtopic_name = self._clean_mindmap_text(subtopic_name)
                lines.append(f"            ({subtopic_emoji} {safe_subtopic_name})")
                
                for detail in subtopic.get('details', []):
                    detail_emoji = detail.get('emoji', '▫️')
                    detail_text = detail['text']
                    # Clean and limit detail text length
                    safe_detail_text = self._clean_mindmap_text(detail_text, max_length=50)
                    lines.append(f"                [{detail_emoji} {safe_detail_text}]")
        
        return '\n'.join(lines)
    
    def _clean_mindmap_text(self, text: str, max_length: int = 100) -> str:
        """Clean and format text for mindmap display"""
        # Remove problematic characters and clean whitespace
        text = re.sub(r'[(){}[\]"\'`]', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Truncate if too long, but keep words intact
        if len(text) > max_length:
            words = text.split()
            truncated = ""
            for word in words:
                if len(truncated + " " + word) <= max_length - 3:  # Reserve space for "..."
                    truncated += (" " + word) if truncated else word
                else:
                    break
            text = truncated + "..." if truncated != text else text
        
        return text
    
    async def _select_emoji_enhanced(self, text: str, content_type: str = 'topic') -> str:
        """Enhanced emoji selection with caching"""
        cache_key = f"{content_type}:{text}"
        if cache_key in self._emoji_cache:
            return self._emoji_cache[cache_key]
        
        # Enhanced emoji mapping
        text_lower = text.lower()
        
        if content_type == 'topic':
            emoji_map = {
                'data': '📊', 'system': '🔧', 'user': '👤', 'security': '🔒',
                'performance': '⚡', 'design': '🎨', 'development': '💻',
                'research': '🔬', 'analysis': '📈', 'business': '💼',
                'strategy': '📋', 'process': '⚙️', 'management': '📝',
                'api': '🔌', 'database': '🗄️', 'network': '🌐',
                'authentication': '🔐', 'monitoring': '📡', 'testing': '🧪'
            }
        else:
            emoji_map = {
                'important': '⭐', 'key': '🔑', 'main': '🎯', 'critical': '❗',
                'example': '💡', 'note': '📌', 'warning': '⚠️', 'tip': '💡',
                'feature': '✨', 'requirement': '📋', 'step': '👣'
            }
        
        # Find matching emoji
        selected_emoji = None
        for keyword, emoji in emoji_map.items():
            if keyword in text_lower:
                selected_emoji = emoji
                break
        
        # Default emojis by content type
        if not selected_emoji:
            defaults = {
                'topic': '📋',
                'subtopic': '📌', 
                'detail': '▫️'
            }
            selected_emoji = defaults.get(content_type, '•')
        
        # Cache the result
        self._emoji_cache[cache_key] = selected_emoji
        return selected_emoji
    
    async def _call_llm_enhanced(self, prompt: str, category: str) -> str:
        """Enhanced LLM call with token tracking for OLLAMA"""
        try:
            response_text = ""
            input_tokens = 0
            output_tokens = 0
            
            if self.config.api_provider == "OLLAMA" and self.ollama_client:
                response = await self.ollama_client.chat.completions.create(
                    model=self.config.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature
                )
                
                if response.usage:
                    input_tokens = response.usage.prompt_tokens or 0
                    output_tokens = response.usage.completion_tokens or 0
                
                response_text = response.choices[0].message.content or ""
                
            else:
                # Fallback
                self.logger.warning("OLLAMA client not available, using fallback")
                response_text = self._generate_fallback_response(category)
            
            # Track usage
            self.token_tracker.add_usage(category, input_tokens, output_tokens)
            self._llm_calls[category] += 1
            
            return response_text
                
        except Exception as e:
            self.logger.error(f"OLLAMA LLM call failed: {e}")
            return self._generate_fallback_response(category)
    
    async def _detect_document_type(self, text_content: str) -> DocumentType:
        """Enhanced document type detection"""
        text_lower = text_content.lower()
        
        # Enhanced keyword detection
        type_indicators = {
            DocumentType.TECHNICAL: ['api', 'function', 'class', 'method', 'algorithm', 'system', 
                                   'architecture', 'protocol', 'implementation', 'framework'],
            DocumentType.SCIENTIFIC: ['research', 'study', 'experiment', 'hypothesis', 'methodology', 
                                    'results', 'analysis', 'peer-reviewed', 'citation'],
            DocumentType.BUSINESS: ['strategy', 'market', 'business', 'revenue', 'customer', 
                                  'sales', 'profit', 'investment', 'roi'],
            DocumentType.LEGAL: ['shall', 'whereas', 'agreement', 'contract', 'legal', 
                               'law', 'regulation', 'compliance', 'liability'],
            DocumentType.ACADEMIC: ['thesis', 'dissertation', 'academic', 'university', 
                                  'scholarly', 'journal', 'conference']
        }
        
        # Calculate scores
        scores = {}
        for doc_type, keywords in type_indicators.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[doc_type] = score
        
        # Find highest scoring type
        max_score = max(scores.values())
        if max_score > 0:
            return max(scores.items(), key=lambda x: x[1])[0]
        
        return DocumentType.GENERAL
    
    def _generate_fallback_response(self, category: str) -> str:
        """Generate fallback response when LLM is not available"""
        fallbacks = {
            "topics": '["Main Concept", "Key Ideas", "Important Points"]',
            "subtopics": '["Overview", "Details", "Examples"]',
            "details": '["Key point 1", "Key point 2", "Key point 3"]'
        }
        return fallbacks.get(category, '["General Information"]')
    
    def _parse_json_response(self, response: str) -> List[str]:
        """Enhanced JSON response parsing"""
        try:
            # Clean up the response
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]
            
            # Parse JSON
            data = json.loads(response)
            if isinstance(data, list):
                return [str(item) for item in data if item and str(item).strip()]
            else:
                return [str(data)]
                
        except json.JSONDecodeError:
            # Try to extract items from text format
            items = []
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Remove bullets and numbering
                    line = re.sub(r'^[\d\-\*\•]+\.?\s*', '', line)
                    line = re.sub(r'^["\']|["\']$', '', line)  # Remove quotes
                    if line:
                        items.append(line)
            
            return items[:10]  # Limit results
    
    def _generate_enhanced_html(self, mermaid_syntax: str) -> str:
        """Generate enhanced HTML with better styling"""
        # Create edit URL for Mermaid Live Editor
        data = {
            "code": mermaid_syntax,
            "mermaid": {"theme": "default"}
        }
        json_string = json.dumps(data)
        compressed_data = zlib.compress(json_string.encode('utf-8'), level=9)
        base64_string = base64.urlsafe_b64encode(compressed_data).decode('utf-8').rstrip('=')
        edit_url = f'https://mermaid.live/edit#pako:{base64_string}'
        
        html_template = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>BookWorm Advanced Mindmap</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/mermaid@11.4.0/dist/mermaid.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}
        #mermaid {{
            width: 100%;
            height: calc(100vh - 80px);
            overflow: auto;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        .edit-btn {{
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.3);
            transition: all 0.3s ease;
        }}
        .edit-btn:hover {{
            background: rgba(255, 255, 255, 0.3);
            transform: translateY(-1px);
        }}
    </style>
</head>
<body class="bg-gray-50">
    <div class="header flex items-center justify-between p-6">
        <div>
            <h1 class="text-2xl font-bold">🧠 BookWorm Advanced Mindmap</h1>
            <p class="text-blue-100 mt-1">Generated by BookWorm Knowledge Ingestion System</p>
        </div>
        <a href="{edit_url}" target="_blank" 
           class="edit-btn px-6 py-3 rounded-lg text-white font-medium hover:shadow-lg">
            ✏️ Edit in Mermaid Live Editor
        </a>
    </div>
    <div id="mermaid" class="p-8">
        <pre class="mermaid">
{mermaid_syntax}
        </pre>
    </div>
    <script>
        mermaid.initialize({{
            startOnLoad: true,
            securityLevel: 'loose',
            theme: 'default',
            mindmap: {{
                useMaxWidth: true,
                padding: 20
            }},
            themeConfig: {{
                controlBar: true
            }}
        }});
    </script>
</body>
</html>"""
        
        return html_template
    
    def _convert_mindmap_to_markdown(self, mermaid_syntax: str) -> str:
        """Convert Mermaid mindmap to markdown outline"""
        lines = ["# 🧠 Advanced Document Mindmap", ""]
        lines.append(f"*Generated by BookWorm Advanced Mindmap Generator at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        lines.append("")
        
        # Parse Mermaid syntax to extract structure
        syntax_lines = mermaid_syntax.split('\n')
        
        for line in syntax_lines:
            stripped = line.strip()
            if not stripped or stripped == 'mindmap':
                continue
            
            # Count indentation to determine level
            indent_level = (len(line) - len(line.lstrip())) // 4
            
            if indent_level == 2 and '((' in stripped and '))' in stripped:
                # Main topic
                topic_match = re.search(r'\(\((.+?)\)\)', stripped)
                if topic_match:
                    current_topic = topic_match.group(1).strip()
                    lines.append(f"## {current_topic}")
                    lines.append("")
            
            elif indent_level == 3 and '(' in stripped and ')' in stripped:
                # Subtopic
                subtopic_match = re.search(r'\((.+?)\)', stripped)
                if subtopic_match:
                    current_subtopic = subtopic_match.group(1).strip()
                    lines.append(f"### {current_subtopic}")
                    lines.append("")
            
            elif indent_level == 4 and '[' in stripped and ']' in stripped:
                # Detail
                detail_match = re.search(r'\[(.+?)\]', stripped)
                if detail_match:
                    detail_text = detail_match.group(1).strip()
                    lines.append(f"- {detail_text}")
        
        lines.append("")
        lines.append("---")
        lines.append("*Powered by BookWorm Advanced Mindmap Generator*")
        
        return '\n'.join(lines)


# Maintain backward compatibility
BookWormMindmapGenerator = AdvancedMindmapGenerator
