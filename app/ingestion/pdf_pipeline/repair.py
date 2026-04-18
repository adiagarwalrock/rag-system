from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

import pymupdf as fitz

logger = logging.getLogger(__name__)


def suppress_mupdf_messages() -> None:
    """Hide noisy raw MuPDF console output and rely on structured app logs.

    Returns:
        None
    """
    try:
        fitz.TOOLS.mupdf_display_errors(False)
        fitz.TOOLS.mupdf_display_warnings(False)
    except Exception:
        logger.exception("Failed to suppress MuPDF warnings/errors")


def repair_pdf_path(file_path: str) -> tuple[str, dict]:
    """Best-effort repair for malformed PDFs before layout parsing.

    Args:
        file_path: Path handling the corrupted PDF.

    Returns:
        A tuple of (repaired filepath or original, repair metadata dict).
    """
    source = Path(file_path)
    status = {
        "pdf_repair_attempted": False,
        "pdf_repair_method": None,
        "pdf_repair_success": False,
        "pdf_repair_error": None,
    }

    tmp_dir = Path(tempfile.mkdtemp(prefix="pdf_repair_"))
    mutool_out = tmp_dir / "repaired_mutool.pdf"
    status["pdf_repair_attempted"] = True
    status["pdf_repair_method"] = "mutool_clean"
    if _try_mutool_clean(source, mutool_out):
        status["pdf_repair_success"] = True
        return str(mutool_out), status

    status["pdf_repair_method"] = "pymupdf_page_copy"
    page_copy_out = tmp_dir / "repaired_page_copy.pdf"
    ok, error = _try_page_copy_rebuild(source, page_copy_out)
    if ok:
        status["pdf_repair_success"] = True
        return str(page_copy_out), status

    status["pdf_repair_error"] = error
    return file_path, status


def _try_mutool_clean(source: Path, output: Path) -> bool:
    """Attempt mutool clean repair process.

    Args:
        source: Source PDF pathlib path.
        output: Destination PDF pathlib path.

    Returns:
        True if successful, False if mutool failed.
    """
    try:
        completed = subprocess.run(
            ["mutool", "clean", str(source), str(output)],
            capture_output=True,
            text=True,
            check=False,
            timeout=90,
        )
    except Exception:
        logger.exception("mutool clean execution failed for %s", source)
        return False

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        logger.warning("mutool clean failed for %s: %s", source, stderr[:500])
        return False
    return output.exists() and output.stat().st_size > 0


def _try_page_copy_rebuild(source: Path, output: Path) -> tuple[bool, str | None]:
    """Attempt a PyMuPDF page-copy rebuild on corrupt files.

    Args:
        source: Source PDF path.
        output: Destination PDF rebuilt path.

    Returns:
        Tuple of (True/False if repair was successful, error string).
    """
    try:
        with fitz.open(str(source)) as src_doc, fitz.open() as dst_doc:
            dst_doc.insert_pdf(src_doc)
            dst_doc.save(str(output), garbage=4, deflate=True)
        if output.exists() and output.stat().st_size > 0:
            return True, None
        return False, "PyMuPDF page-copy output missing or empty"
    except Exception as exc:
        logger.warning("PyMuPDF page-copy repair failed for %s: %s", source, exc)
        return False, str(exc)
