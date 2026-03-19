"""
Resume Parser: Extracts text from PDF and structures it using LLM.
"""
import json
import re
from pathlib import Path
from typing import Optional, Dict, Any

import fitz  # PyMuPDF
import pdfplumber
from loguru import logger
from openai import OpenAI

from utils.prompts import RESUME_PARSER_SYSTEM, RESUME_PARSER_PROMPT
from config import settings


def extract_text_pymupdf(pdf_path: str) -> str:
    """Extract text from PDF using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        doc.close()
        return text.strip()
    except Exception as e:
        logger.warning(f"PyMuPDF extraction failed: {e}, trying pdfplumber...")
        return ""


def extract_text_pdfplumber(pdf_path: str) -> str:
    """Extract text from PDF using pdfplumber (fallback)."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"pdfplumber extraction also failed: {e}")
        return ""


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from PDF, with fallback from PyMuPDF → pdfplumber.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text string
    """
    text = extract_text_pymupdf(pdf_path)
    if not text or len(text) < 100:
        logger.info("Falling back to pdfplumber for text extraction")
        text = extract_text_pdfplumber(pdf_path)

    if not text:
        raise ValueError("Could not extract text from PDF. Ensure it's not image-only.")

    logger.info(f"Extracted {len(text)} characters from resume PDF")
    return text


def extract_text_from_bytes(pdf_bytes: bytes, filename: str = "resume.pdf") -> str:
    """
    Extract text from PDF bytes (for uploaded files).
    
    Args:
        pdf_bytes: Raw PDF bytes
        filename: Original filename for logging
        
    Returns:
        Extracted text
    """
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        text = extract_text_from_pdf(tmp_path)
    finally:
        os.unlink(tmp_path)

    return text


def parse_resume_with_llm(resume_text: str) -> Dict[str, Any]:
    """
    Use LLM to parse raw resume text into structured format.
    
    Args:
        resume_text: Raw text extracted from resume
        
    Returns:
        Structured resume dict
    """
    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    prompt = RESUME_PARSER_PROMPT.format(resume_text=resume_text[:6000])  # Cap tokens

    try:
        response = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": RESUME_PARSER_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)
        logger.info(f"Successfully parsed resume for: {parsed.get('name', 'Unknown')}")
        return parsed

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM JSON response: {e}")
        # Fallback: return minimal structure
        return _fallback_parse(resume_text)
    except Exception as e:
        logger.error(f"LLM resume parsing error: {e}")
        return _fallback_parse(resume_text)


def _fallback_parse(text: str) -> Dict[str, Any]:
    """Simple regex-based fallback parser."""
    skills = []
    # Basic skill extraction
    skill_keywords = [
        "Python", "Java", "SQL", "Machine Learning", "Deep Learning",
        "TensorFlow", "PyTorch", "Scikit-learn", "AWS", "Docker",
        "Kubernetes", "FastAPI", "Django", "React", "Node.js",
        "NLP", "Computer Vision", "Pandas", "NumPy", "Spark",
    ]
    for skill in skill_keywords:
        if skill.lower() in text.lower():
            skills.append(skill)

    return {
        "name": "Unknown",
        "email": None,
        "phone": None,
        "skills": skills,
        "programming_languages": [],
        "frameworks": [],
        "experience": [],
        "projects": [],
        "education": [],
        "certifications": [],
        "summary": text[:500],
    }


def build_resume_context(parsed_resume: Dict[str, Any]) -> str:
    """
    Build a concise context string from parsed resume for use in prompts.
    
    Args:
        parsed_resume: Structured resume dict
        
    Returns:
        Context string for LLM prompts
    """
    parts = []

    if parsed_resume.get("name") and parsed_resume["name"] != "Unknown":
        parts.append(f"Candidate: {parsed_resume['name']}")

    if parsed_resume.get("summary"):
        parts.append(f"Summary: {parsed_resume['summary']}")

    if parsed_resume.get("skills"):
        parts.append(f"Skills: {', '.join(parsed_resume['skills'][:15])}")

    if parsed_resume.get("programming_languages"):
        parts.append(f"Languages: {', '.join(parsed_resume['programming_languages'])}")

    if parsed_resume.get("frameworks"):
        parts.append(f"Frameworks: {', '.join(parsed_resume['frameworks'])}")

    if parsed_resume.get("experience"):
        exp_lines = []
        for exp in parsed_resume["experience"][:3]:
            highlights = "; ".join(exp.get("highlights", [])[:2])
            exp_lines.append(
                f"- {exp.get('role', '')} at {exp.get('company', '')} ({exp.get('duration', '')}): {highlights}"
            )
        parts.append("Experience:\n" + "\n".join(exp_lines))

    if parsed_resume.get("projects"):
        proj_lines = []
        for proj in parsed_resume["projects"][:3]:
            techs = ", ".join(proj.get("technologies", [])[:4])
            key = "; ".join(proj.get("key_aspects", [])[:2])
            proj_lines.append(f"- {proj.get('name', '')}: {proj.get('description', '')} [{techs}] — {key}")
        parts.append("Projects:\n" + "\n".join(proj_lines))

    return "\n".join(parts) if parts else "No resume provided."


class ResumeParser:
    """Main interface for resume parsing operations."""

    def __init__(self):
        self.parsed_data: Optional[Dict[str, Any]] = None
        self.raw_text: str = ""

    def parse_from_path(self, pdf_path: str) -> Dict[str, Any]:
        """Parse resume from file path."""
        self.raw_text = extract_text_from_pdf(pdf_path)
        self.parsed_data = parse_resume_with_llm(self.raw_text)
        return self.parsed_data

    def parse_from_bytes(self, pdf_bytes: bytes) -> Dict[str, Any]:
        """Parse resume from raw bytes."""
        self.raw_text = extract_text_from_bytes(pdf_bytes)
        self.parsed_data = parse_resume_with_llm(self.raw_text)
        return self.parsed_data

    def get_context(self) -> str:
        """Get formatted context string for use in prompts."""
        if not self.parsed_data:
            return "No resume provided."
        return build_resume_context(self.parsed_data)

    def get_raw_text(self) -> str:
        return self.raw_text