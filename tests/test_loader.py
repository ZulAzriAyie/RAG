"""
tests/test_loader.py
Unit tests for document indexing and chunking logic.
"""

from unittest.mock import MagicMock, patch
from src.indexing.loader import extract_metadata_and_chunks


# Fix: Match the loader class used in src/indexing/loader.py (PyMuPDFLoader)
@patch("src.indexing.loader.PyMuPDFLoader")
def test_extract_metadata_and_chunks(mock_pdf_loader, tmp_path):
  """Mock PDF loading to test output formatting and chunk splitting without physical PDFs."""
  fake_pdf = tmp_path / "dummy.pdf"
  fake_pdf.write_text("Dummy binary content")

  # Mock document chunks returned by the loader
  mock_doc = MagicMock()
  mock_doc.page_content = (
      "Abstract: This research introduces advanced agent architectures."
  )
  mock_doc.metadata = {"page": 0}

  mock_instance = mock_pdf_loader.return_value
  mock_instance.load.return_value = [mock_doc]

  metadata_profile, chunks = extract_metadata_and_chunks(fake_pdf)

  assert chunks is not None
  assert len(chunks) > 0
  assert metadata_profile is not None