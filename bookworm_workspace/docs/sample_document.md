# BookWorm Demo Document

## Introduction

This is a sample document to demonstrate the BookWorm knowledge graph system. BookWorm is designed to process documents, extract knowledge, and generate mind maps.

## Key Features

### Document Processing
- **PDF Support**: Uses MinerU for advanced PDF processing with layout preservation
- **Multiple Formats**: Supports PDF, TXT, MD, DOCX, and PPTX files
- **Smart Extraction**: Maintains document structure and relationships

### Knowledge Graph
- **LightRAG Integration**: Uses LightRAG for building knowledge graphs from documents
- **Ollama Support**: Works with local Ollama models for privacy
- **Multiple Query Modes**: Supports local, global, hybrid, and naive query modes

### Mind Map Generation
- **AI-Powered**: Uses LLMs to generate structured mind maps
- **Multiple Providers**: Supports OpenAI, Claude, DeepSeek, and Gemini
- **Visual Output**: Creates interactive mind maps from document knowledge

## Technical Architecture

The system consists of three main components:

1. **DocumentProcessor**: Handles file ingestion and conversion
2. **KnowledgeGraph**: Manages the LightRAG-based knowledge storage
3. **MindmapGenerator**: Creates visual representations of knowledge

## Use Cases

- **Research**: Process academic papers and generate knowledge maps
- **Documentation**: Convert technical docs into searchable knowledge graphs
- **Learning**: Create study materials from textbooks and articles
- **Analysis**: Extract insights from large document collections

## Getting Started

To use BookWorm:

1. Configure your environment variables
2. Place documents in the docs directory
3. Run the processing pipeline
4. Query the knowledge graph
5. Generate mind maps from results

## Conclusion

BookWorm provides a complete pipeline from document ingestion to knowledge visualization, making it easy to extract and organize information from large document collections.
