"""
Document chunking strategies for RAG pipeline.
"""
import re
from typing import List, Dict, Any
from dataclasses import dataclass
from loguru import logger


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    chunk_id: int
    source: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?;:()\-\n]', '', text)
    return text.strip()


def chunk_by_sentences(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    source: str = "document"
) -> List[Chunk]:
    """
    Chunk text by sentences with overlap.
    
    Args:
        text: Input text
        chunk_size: Target characters per chunk
        overlap: Character overlap between chunks
        source: Source document name
        
    Returns:
        List of Chunk objects
    """
    # Split by sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    chunks = []
    current_chunk = ""
    chunk_id = 0

    for sentence in sentences:
        # If adding this sentence would exceed chunk_size
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(Chunk(
                text=current_chunk.strip(),
                chunk_id=chunk_id,
                source=source,
                metadata={"strategy": "sentence", "char_count": len(current_chunk)}
            ))
            chunk_id += 1
            # Start new chunk with overlap
            words = current_chunk.split()
            overlap_words = words[max(0, len(words) - 20):]
            current_chunk = " ".join(overlap_words) + " " + sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence

    # Add remaining text as final chunk
    if current_chunk.strip():
        chunks.append(Chunk(
            text=current_chunk.strip(),
            chunk_id=chunk_id,
            source=source,
            metadata={"strategy": "sentence", "char_count": len(current_chunk)}
        ))

    logger.info(f"Created {len(chunks)} chunks from {source} (strategy: sentence)")
    return chunks


def chunk_by_fixed_size(
    text: str,
    chunk_size: int = 400,
    overlap: int = 80,
    source: str = "document"
) -> List[Chunk]:
    """
    Fixed-size chunking with overlap (word-boundary aware).
    
    Args:
        text: Input text
        chunk_size: Words per chunk
        overlap: Words to overlap
        source: Source name
        
    Returns:
        List of Chunk objects
    """
    words = text.split()
    chunks = []
    chunk_id = 0

    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        if len(chunk_text.strip()) > 20:  # Filter tiny chunks
            chunks.append(Chunk(
                text=chunk_text.strip(),
                chunk_id=chunk_id,
                source=source,
                metadata={"strategy": "fixed", "word_count": len(chunk_words)}
            ))
            chunk_id += 1

        start += chunk_size - overlap

    logger.info(f"Created {len(chunks)} chunks from {source} (strategy: fixed-size)")
    return chunks


def chunk_by_sections(
    text: str,
    source: str = "document"
) -> List[Chunk]:
    """
    Chunk by document sections (headers, bullet points).
    Good for structured documents like JDs.
    
    Args:
        text: Input text
        source: Source name
        
    Returns:
        List of Chunk objects
    """
    # Split on section headers (ALL CAPS lines, numbered sections, bullet-heavy areas)
    section_pattern = r'\n(?=[A-Z][A-Z\s]{3,}:|\d+\.\s|•\s|-\s)'
    sections = re.split(section_pattern, text)

    chunks = []
    chunk_id = 0

    for section in sections:
        section = section.strip()
        if not section or len(section) < 30:
            continue

        # If section is too long, sub-chunk it
        if len(section) > 600:
            sub_chunks = chunk_by_sentences(section, chunk_size=400, source=source)
            for sub in sub_chunks:
                sub.chunk_id = chunk_id
                chunks.append(sub)
                chunk_id += 1
        else:
            chunks.append(Chunk(
                text=section,
                chunk_id=chunk_id,
                source=source,
                metadata={"strategy": "section"}
            ))
            chunk_id += 1

    if not chunks:
        # Fallback to sentence chunking
        return chunk_by_sentences(text, source=source)

    logger.info(f"Created {len(chunks)} chunks from {source} (strategy: sections)")
    return chunks


def smart_chunk_jd(jd_text: str) -> List[Chunk]:
    """
    Smart chunking specifically for Job Descriptions.
    Uses section-aware chunking for better retrieval.
    
    Args:
        jd_text: Job description text
        
    Returns:
        List of Chunk objects
    """
    cleaned = clean_text(jd_text)
    chunks = chunk_by_sections(cleaned, source="job_description")

    # If very short JD, just use as single chunk
    if len(chunks) == 0:
        return [Chunk(text=cleaned, chunk_id=0, source="job_description")]

    return chunks


def chunk_text(
    text: str,
    source: str = "document",
    strategy: str = "auto"
) -> List[Chunk]:
    """
    Main chunking interface. Selects strategy automatically.
    
    Args:
        text: Text to chunk
        source: Source name
        strategy: 'auto', 'sentence', 'fixed', 'section'
        
    Returns:
        List of Chunk objects
    """
    text = clean_text(text)

    if strategy == "sentence":
        return chunk_by_sentences(text, source=source)
    elif strategy == "fixed":
        return chunk_by_fixed_size(text, source=source)
    elif strategy == "section":
        return chunk_by_sections(text, source=source)
    else:
        # Auto: use section chunking for JDs, sentence for resumes
        if "job description" in source.lower() or "jd" in source.lower():
            return chunk_by_sections(text, source=source)
        else:
            return chunk_by_sentences(text, source=source)