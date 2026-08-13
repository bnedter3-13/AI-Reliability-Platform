"""
scripts/index_pdf_demo.py

Knowledge Base + Retrieval Verification Agent (Component 11) demo/manual-test
script: generates a small real PDF (via PyMuPDF) with known factual content,
indexes it through the full extract -> chunk -> embed -> store pipeline, and
prints the result so it can be checked by hand.

Usage:
    python scripts/index_pdf_demo.py [pdf_path] [project_id]

If pdf_path is omitted, a sample PDF is generated in a temp directory.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fitz

from app.knowledge_base.indexing_service import index_pdf

SAMPLE_TEXT = (
    "The AI Reliability Platform was founded in 2026 by Team Nexus AI as a SIC "
    "AI Capstone project. The platform's headquarters is located in Athens, "
    "Greece. The founding team built the platform to monitor, evaluate, and "
    "improve the reliability of AI and RAG systems running in production "
    "environments. The AI Reliability Platform is headquartered in Athens, "
    "Greece.\n\n"
    "The AI Reliability Platform's refund policy is straightforward: customers "
    "who purchase the paid tier may request a full refund within 30 days of "
    "purchase, no questions asked. Refund requests can be submitted through the "
    "billing portal or by emailing support. Once a refund request within the "
    "30 day refund window is approved, the refund is processed back to the "
    "original payment method within 5 to 7 business days.\n\n"
    "The platform's Core Evaluation component uses an LLM-as-a-judge approach "
    "to score correctness, faithfulness, and hallucination risk for every "
    "answer produced by an AI system under test. When an evaluation fails, the "
    "Root Cause Analysis component classifies why it failed using rule-based "
    "signals combined with the judge's own explanation text."
)


def _generate_sample_pdf() -> str:
    doc = fitz.open()
    page = doc.new_page()
    # insert_textbox wraps text within the rect; insert_text (single position) doesn't
    # wrap and silently truncates anything past the page width.
    page.insert_textbox(fitz.Rect(72, 72, page.rect.width - 72, page.rect.height - 72), SAMPLE_TEXT, fontsize=11)
    path = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False).name
    doc.save(path)
    doc.close()
    return path


def main() -> None:
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else _generate_sample_pdf()
    project_id = sys.argv[2] if len(sys.argv) > 2 else "kb-demo"

    print(f"Indexing '{pdf_path}' into project '{project_id}'...")
    result = index_pdf(project_id=project_id, file_path=pdf_path, original_filename=pdf_path)

    print(f"doc_id:     {result.doc_id}")
    print(f"num_chunks: {result.num_chunks}")


if __name__ == "__main__":
    main()
