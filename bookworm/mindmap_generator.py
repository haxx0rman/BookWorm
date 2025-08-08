"""
Integrated Mindmap Generator for BookWorm
Based on the Dicklesworthstone mindmap-generator with BookWorm integration
"""
import asyncio
import json
import logging
import re
import time
import hashlib
import base64
import zlib
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

# Optional imports with fallbacks
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

try:
    from anthropic import AsyncAnthropic
except ImportError:
    AsyncAnthropic = None

try:
    from google import genai
except ImportError:
    genai = None

try:
    from fuzzywuzzy import fuzz
except ImportError:
    fuzz = None

try:
    from termcolor import colored
except ImportError:
    def colored(text, color=None, attrs=None):
        return text

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


class BookWormMindmapGenerator:
    """
    Integrated mindmap generator for BookWorm system
    Combines LightRAG knowledge extraction with hierarchical mindmap generation
    """
    
    def __init__(self, config: BookWormConfig):
        self.config = config
        self.logger = logging.getLogger("bookworm.mindmap_generator")
        
        # Initialize LLM clients based on provider
        self._init_llm_clients()
        
        # Mindmap generation settings
        self.max_topics = 6
        self.max_subtopics_per_topic = 4
        self.max_details_per_subtopic = 8
        
        # Token usage tracking
        self.token_usage = TokenUsage(provider=config.api_provider)
        
        # Caching for efficiency
        self._content_cache = {}
        self._emoji_cache = {}
        
        # Initialize document type prompts
        self._init_prompts()
    
    def _init_llm_clients(self):
        """Initialize LLM clients based on configuration"""
        self.openai_client = None
        self.anthropic_client = None
        self.deepseek_client = None
        self.gemini_client = None
        
        if self.config.api_provider == "OPENAI" and self.config.openai_api_key:
            if AsyncOpenAI:
                self.openai_client = AsyncOpenAI(api_key=self.config.openai_api_key)
        
        elif self.config.api_provider == "CLAUDE" and self.config.anthropic_api_key:
            if AsyncAnthropic:
                self.anthropic_client = AsyncAnthropic(api_key=self.config.anthropic_api_key)
        
        elif self.config.api_provider == "DEEPSEEK" and self.config.deepseek_api_key:
            if AsyncOpenAI:
                self.deepseek_client = AsyncOpenAI(
                    api_key=self.config.deepseek_api_key,
                    base_url="https://api.deepseek.com"
                )
        
        elif self.config.api_provider == "GEMINI" and self.config.gemini_api_key:
            if genai:
                self.gemini_client = genai.Client(
                    api_key=self.config.gemini_api_key,
                    http_options={"api_version": "v1alpha"}
                )
    
    def _init_prompts(self):
        """Initialize document type-specific prompts"""
        self.type_prompts = {
            DocumentType.TECHNICAL: {
                'topics': """Analyze this technical document focusing on core system components and relationships.
                
Identify major architectural or technical components that form complete, independent units of functionality.
Each component should be:
- A distinct technical system, module, or process
- Independent enough to be understood on its own
- Critical to the overall system functionality
- Connected to at least one other component

Consider:
1. What are the core building blocks?
2. How do these pieces fit together?
3. What dependencies exist between components?
4. What are the key technical boundaries?

Format: Return a JSON array of component names that represent the highest-level technical building blocks.""",
                
                'subtopics': """For the technical component '{topic}', identify its essential sub-components and interfaces.

Each subtopic should:
- Represent a crucial aspect of this component
- Have clear technical responsibilities
- Interface with other parts of the system
- Contribute to the component's core purpose

Consider:
1. What interfaces does this component expose?
2. What are its internal subsystems?
3. How does it process data or handle requests?
4. What services does it provide to other components?
5. What technical standards or protocols does it implement?

Format: Return a JSON array of technical subtopic names that form this component's architecture.""",
                
                'details': """For the technical subtopic '{subtopic}', identify specific implementation aspects and requirements.

Focus on:
1. Key algorithms or methods
2. Data structures and formats
3. Protocol specifications
4. Performance characteristics
5. Error handling approaches
6. Security considerations
7. Dependencies and requirements

Include concrete technical details that are:
- Implementation-specific
- Measurable or testable
- Critical for understanding
- Relevant to integration

Format: Return a JSON array of technical specifications and implementation details."""
            },
            
            DocumentType.GENERAL: {
                'topics': """Analyze this document focusing on main conceptual themes and relationships.

Identify major themes that:
- Represent complete, independent ideas
- Form logical groupings of related concepts
- Support the document's main purpose
- Connect to other important themes

Consider:
1. What are the fundamental ideas being presented?
2. How do these ideas relate to each other?
3. What are the key areas of focus?
4. What concepts appear most frequently or prominently?

Avoid topics that are:
- Too narrow (specific examples without broader context)
- Too broad (encompassing the entire document)
- Purely administrative or formatting elements
- Isolated without connections to other concepts

Format: Return a JSON array of main conceptual themes.""",
                
                'subtopics': """For the theme '{topic}', identify key supporting concepts and relationships.

Each subtopic should:
- Directly support or elaborate on the main theme
- Represent a distinct aspect or dimension
- Connect to other subtopics within this theme
- Contribute meaningful content to understanding

Consider:
1. What specific aspects of this theme are discussed?
2. What examples or evidence support this theme?
3. What different perspectives or approaches are presented?
4. How does this theme connect to practical applications?

Format: Return a JSON array of supporting concepts that develop this theme.""",
                
                'details': """For the subtopic '{subtopic}', extract specific information and supporting details.

Focus on:
1. Concrete examples or case studies
2. Specific data, statistics, or measurements
3. Key quotes or important statements
4. Procedural steps or methodologies
5. Supporting evidence or research
6. Practical applications or implications
7. Important definitions or clarifications

Include details that are:
- Factually specific
- Directly relevant to the subtopic
- Informative and substantive
- Representative of the source content

Format: Return a JSON array of specific details and supporting information."""
            }
        }
    
    async def generate_mindmap_from_text(self, text_content: str, document_id: str) -> MindmapGenerationResult:
        """Generate a comprehensive mindmap from text content"""
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting mindmap generation for document {document_id}")
            
            # Detect document type
            doc_type = await self._detect_document_type(text_content)
            self.logger.info(f"Detected document type: {doc_type.value}")
            
            # Extract hierarchical content
            topics = await self._extract_topics(text_content, doc_type)
            self.logger.info(f"Extracted {len(topics)} main topics")
            
            # Process each topic to get subtopics and details
            processed_topics = []
            for topic in topics:
                processed_topic = await self._process_topic(topic, text_content, doc_type)
                processed_topics.append(processed_topic)
            
            # Generate mindmap syntax
            mermaid_syntax = await self._generate_mermaid_syntax(processed_topics, document_id)
            
            # Generate HTML and markdown
            html_content = self._generate_html_content(mermaid_syntax)
            markdown_outline = self._generate_markdown_outline(processed_topics)
            
            processing_time = time.time() - start_time
            
            result = MindmapGenerationResult(
                document_id=document_id,
                mermaid_syntax=mermaid_syntax,
                html_content=html_content,
                markdown_outline=markdown_outline,
                token_usage=self.token_usage,
                processing_time=processing_time
            )
            
            self.logger.info(f"Mindmap generation completed in {processing_time:.2f} seconds")
            return result
            
        except Exception as e:
            self.logger.error(f"Mindmap generation failed: {e}")
            raise MindMapGenerationError(f"Failed to generate mindmap: {e}")
    
    async def _detect_document_type(self, text_content: str) -> DocumentType:
        """Detect the type of document for optimized processing"""
        # Simple heuristic-based detection
        text_lower = text_content.lower()
        
        # Technical indicators
        technical_keywords = ['api', 'function', 'class', 'method', 'algorithm', 'system', 'architecture', 'protocol']
        
        # Scientific indicators
        scientific_keywords = ['research', 'study', 'experiment', 'hypothesis', 'methodology', 'results', 'analysis']
        
        # Business indicators
        business_keywords = ['strategy', 'market', 'business', 'revenue', 'customer', 'sales', 'profit']
        
        # Legal indicators
        legal_keywords = ['shall', 'whereas', 'agreement', 'contract', 'legal', 'law', 'regulation']
        
        # Count keyword occurrences
        tech_score = sum(1 for keyword in technical_keywords if keyword in text_lower)
        sci_score = sum(1 for keyword in scientific_keywords if keyword in text_lower)
        biz_score = sum(1 for keyword in business_keywords if keyword in text_lower)
        legal_score = sum(1 for keyword in legal_keywords if keyword in text_lower)
        
        # Determine type based on highest score
        scores = {
            DocumentType.TECHNICAL: tech_score,
            DocumentType.SCIENTIFIC: sci_score,
            DocumentType.BUSINESS: biz_score,
            DocumentType.LEGAL: legal_score,
        }
        
        max_score = max(scores.values())
        if max_score > 0:
            return max(scores.items(), key=lambda x: x[1])[0]
        
        return DocumentType.GENERAL
    
    async def _extract_topics(self, text_content: str, doc_type: DocumentType) -> List[Dict[str, Any]]:
        """Extract main topics from the document"""
        prompt = self.type_prompts[doc_type]['topics']
        
        # Truncate content if too long
        max_content_length = 4000
        if len(text_content) > max_content_length:
            text_content = text_content[:max_content_length] + "..."
        
        full_prompt = f"{prompt}\n\nDocument content:\n{text_content}"
        
        response = await self._call_llm(full_prompt, "topic_extraction")
        topics_data = self._parse_json_response(response)
        
        # Convert to structured format
        topics = []
        for i, topic_name in enumerate(topics_data[:self.max_topics]):
            if isinstance(topic_name, str):
                topics.append({
                    'name': topic_name.strip(),
                    'emoji': await self._select_emoji(topic_name),
                    'subtopics': []
                })
        
        return topics
    
    async def _process_topic(self, topic: Dict[str, Any], text_content: str, doc_type: DocumentType) -> Dict[str, Any]:
        """Process a topic to extract subtopics and details"""
        topic_name = topic['name']
        
        # Extract subtopics
        subtopics_prompt = self.type_prompts[doc_type]['subtopics'].format(topic=topic_name)
        full_prompt = f"{subtopics_prompt}\n\nDocument content:\n{text_content[:3000]}"
        
        subtopics_response = await self._call_llm(full_prompt, "subtopic_extraction")
        subtopics_data = self._parse_json_response(subtopics_response)
        
        # Process subtopics
        processed_subtopics = []
        for subtopic_name in subtopics_data[:self.max_subtopics_per_topic]:
            if isinstance(subtopic_name, str):
                # Extract details for this subtopic
                details = await self._extract_details(subtopic_name, text_content, doc_type)
                
                processed_subtopics.append({
                    'name': subtopic_name.strip(),
                    'emoji': await self._select_emoji(subtopic_name),
                    'details': details
                })
        
        topic['subtopics'] = processed_subtopics
        return topic
    
    async def _extract_details(self, subtopic_name: str, text_content: str, doc_type: DocumentType) -> List[Dict[str, Any]]:
        """Extract details for a specific subtopic"""
        details_prompt = self.type_prompts[doc_type]['details'].format(subtopic=subtopic_name)
        full_prompt = f"{details_prompt}\n\nDocument content:\n{text_content[:2000]}"
        
        details_response = await self._call_llm(full_prompt, "detail_extraction")
        details_data = self._parse_json_response(details_response)
        
        # Process details
        processed_details = []
        for detail in details_data[:self.max_details_per_subtopic]:
            if isinstance(detail, str):
                processed_details.append({
                    'text': detail.strip(),
                    'emoji': await self._select_emoji(detail, content_type='detail')
                })
        
        return processed_details
    
    async def _select_emoji(self, text: str, content_type: str = 'topic') -> str:
        """Select an appropriate emoji for the given text"""
        # Simple emoji selection based on keywords
        text_lower = text.lower()
        
        # Topic-level emojis
        if content_type == 'topic':
            emoji_map = {
                'data': '📊', 'system': '🔧', 'user': '👤', 'security': '🔒',
                'performance': '⚡', 'design': '🎨', 'development': '💻',
                'research': '🔬', 'analysis': '📈', 'business': '💼',
                'strategy': '📋', 'process': '⚙️', 'management': '📝'
            }
        else:
            # Detail-level emojis
            emoji_map = {
                'important': '⭐', 'key': '🔑', 'main': '🎯', 'critical': '❗',
                'example': '💡', 'note': '📌', 'warning': '⚠️', 'tip': '💡'
            }
        
        # Find matching emoji
        for keyword, emoji in emoji_map.items():
            if keyword in text_lower:
                return emoji
        
        # Default emojis by content type
        defaults = {
            'topic': '📋',
            'subtopic': '📌', 
            'detail': '▫️'
        }
        
        return defaults.get(content_type, '•')
    
    async def _call_llm(self, prompt: str, task: str) -> str:
        """Call the configured LLM with the given prompt"""
        try:
            if self.config.api_provider == "OPENAI" and self.openai_client:
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    temperature=0.7
                )
                
                # Track token usage
                self.token_usage.input_tokens += response.usage.prompt_tokens
                self.token_usage.output_tokens += response.usage.completion_tokens
                
                return response.choices[0].message.content
                
            elif self.config.api_provider == "CLAUDE" and self.anthropic_client:
                message = await self.anthropic_client.messages.create(
                    model="claude-3-5-haiku-latest",
                    max_tokens=2000,
                    temperature=0.7,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                # Track token usage
                self.token_usage.input_tokens += message.usage.input_tokens
                self.token_usage.output_tokens += message.usage.output_tokens
                
                return message.content[0].text
                
            else:
                # Fallback - generate simple structured response
                self.logger.warning(f"No LLM client available for {self.config.api_provider}, using fallback")
                return self._generate_fallback_response(task)
                
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            return self._generate_fallback_response(task)
    
    def _generate_fallback_response(self, task: str) -> str:
        """Generate a fallback response when LLM is not available"""
        if task == "topic_extraction":
            return '["Main Concept", "Key Ideas", "Important Points"]'
        elif task == "subtopic_extraction":
            return '["Overview", "Details", "Examples"]'
        elif task == "detail_extraction":
            return '["Key point 1", "Key point 2", "Key point 3"]'
        else:
            return '["General Information"]'
    
    def _parse_json_response(self, response: str) -> List[str]:
        """Parse JSON response from LLM"""
        try:
            # Clean up the response
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.endswith('```'):
                response = response[:-3]
            
            # Parse JSON
            data = json.loads(response)
            if isinstance(data, list):
                return [str(item) for item in data if item]
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
                    if line:
                        items.append(line)
            
            return items[:10]  # Limit results
    
    async def _generate_mermaid_syntax(self, topics: List[Dict[str, Any]], document_id: str) -> str:
        """Generate Mermaid mindmap syntax"""
        lines = ["mindmap"]
        lines.append(f"    ((📄 Document))")
        
        for topic in topics:
            topic_emoji = topic.get('emoji', '📋')
            topic_name = topic['name']
            lines.append(f"        (({topic_emoji} {topic_name}))")
            
            for subtopic in topic.get('subtopics', []):
                subtopic_emoji = subtopic.get('emoji', '📌')
                subtopic_name = subtopic['name']
                lines.append(f"            ({subtopic_emoji} {subtopic_name})")
                
                for detail in subtopic.get('details', []):
                    detail_emoji = detail.get('emoji', '▫️')
                    detail_text = detail['text'][:80]  # Truncate long details
                    lines.append(f"                [{detail_emoji} {detail_text}]")
        
        return '\n'.join(lines)
    
    def _generate_html_content(self, mermaid_syntax: str) -> str:
        """Generate HTML content with Mermaid visualization"""
        # Encode for Mermaid Live Editor
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
    <title>BookWorm Mindmap</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>
        body {{ margin: 0; padding: 20px; font-family: Arial, sans-serif; }}
        .header {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .edit-link {{ color: #0066cc; text-decoration: none; }}
        .edit-link:hover {{ text-decoration: underline; }}
        #mermaid {{ background: white; border: 1px solid #ddd; border-radius: 5px; padding: 20px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📄 BookWorm Mindmap</h1>
        <p>Generated by BookWorm Knowledge Ingestion System</p>
        <p><a href="{edit_url}" target="_blank" class="edit-link">✏️ Edit in Mermaid Live Editor</a></p>
    </div>
    <div id="mermaid">
        <pre class="mermaid">
{mermaid_syntax}
        </pre>
    </div>
    <script>
        mermaid.initialize({{
            startOnLoad: true,
            theme: 'default',
            mindmap: {{ useMaxWidth: true }},
            securityLevel: 'loose'
        }});
    </script>
</body>
</html>"""
        
        return html_template
    
    def _generate_markdown_outline(self, topics: List[Dict[str, Any]]) -> str:
        """Generate markdown outline from topics"""
        lines = ["# Document Mindmap Outline", ""]
        lines.append(f"*Generated by BookWorm at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        lines.append("")
        
        for topic in topics:
            topic_emoji = topic.get('emoji', '📋')
            topic_name = topic['name']
            lines.append(f"## {topic_emoji} {topic_name}")
            lines.append("")
            
            for subtopic in topic.get('subtopics', []):
                subtopic_emoji = subtopic.get('emoji', '📌')
                subtopic_name = subtopic['name']
                lines.append(f"### {subtopic_emoji} {subtopic_name}")
                lines.append("")
                
                for detail in subtopic.get('details', []):
                    detail_emoji = detail.get('emoji', '▫️')
                    detail_text = detail['text']
                    lines.append(f"- {detail_emoji} {detail_text}")
                
                lines.append("")
        
        return '\n'.join(lines)
