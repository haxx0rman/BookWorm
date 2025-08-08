#!/usr/bin/env python3
"""
Simple test script for chunking functionality
"""
import os
import asyncio
from bookworm.mindmap_generator import AdvancedMindmapGenerator, DocumentType
from bookworm.utils import BookWormConfig

async def test_chunking():
    # Set environment variables for more aggressive chunking
    os.environ['BOOKWORM_CHUNK_MAX_WORDS'] = '500'  # Very small threshold
    os.environ['BOOKWORM_CHUNK_MAX_CHARS'] = '3000'  # Very small threshold
    os.environ['BOOKWORM_CHUNK_SIZE'] = '300'
    os.environ['BOOKWORM_CHUNK_OVERLAP'] = '50'
    
    print("🔧 Testing chunking functionality...")
    
    # Create generator
    config = BookWormConfig()
    generator = AdvancedMindmapGenerator(config)
    
    # Read our test document
    test_file = "bookworm_workspace/docs/chunking_test_doc.md"
    with open(test_file, 'r') as f:
        content = f.read()
    
    print(f"📄 Document: {len(content.split())} words, {len(content)} chars")
    
    # Test if chunking is needed
    needs_chunking = generator._needs_chunking(content)
    print(f"🤔 Needs chunking: {needs_chunking}")
    
    if needs_chunking:
        print("📦 Testing chunk creation...")
        chunks = generator._create_chunks(content)
        print(f"✅ Created {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i+1}: {chunk['word_count']} words")
            
        print("🗺️  Testing chunked mindmap generation...")
        try:
            # Test the chunked mindmap generation
            result = await generator._generate_chunked_mindmap(content, DocumentType.TECHNICAL, "test_doc")
            print("✅ Chunked mindmap generation completed!")
            print(f"📊 Generated mindmap length: {len(result)} characters")
        except Exception as e:
            print(f"❌ Chunked mindmap generation failed: {e}")
    
    print("✅ Test completed!")

if __name__ == "__main__":
    asyncio.run(test_chunking())
