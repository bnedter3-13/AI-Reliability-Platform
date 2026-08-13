"""
app/knowledge_base/document_loader.py

Knowledge Base + Retrieval Verification Agent (Component 11): PDF text extraction.
"""

import fitz  # PyMuPDF


def load_pdf_text(file_path: str) -> str:
    """Extract and concatenate the text of every page in a PDF."""
    with fitz.open(file_path) as doc:
        return "\n".join(page.get_text() for page in doc)
