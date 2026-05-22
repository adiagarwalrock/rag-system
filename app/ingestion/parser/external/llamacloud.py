import time
from pathlib import Path


def run(pdf_path: Path) -> tuple[str, float]:
    """Parse a document with LlamaParse; return (markdown, elapsed_seconds)."""
    from llama_parse import LlamaParse  # soft import — only needed when flag is on

    from app.core.config import settings

    start = time.perf_counter()
    parser = LlamaParse(
        api_key=settings.LLAMAPARSE_API_KEY,
        result_type="markdown",
    )
    docs = parser.load_data(str(pdf_path))
    elapsed = time.perf_counter() - start
    return "\n\n".join(d.text for d in docs if d.text), elapsed
