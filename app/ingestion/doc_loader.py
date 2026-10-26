
from pathlib import Path
from tempfile import TemporaryDirectory

import win32com.client

from app.ingestion.docx_loader import load_docx


def load_doc(file_path: str | Path) -> str:
    """Convert a legacy .doc file to .docx and extract its text."""
    path = Path(file_path)

    if path.suffix.lower() != ".doc":
        raise ValueError(
            f"Expected a .doc file, got: {path.name}"
        )

    if not path.is_file():
        raise FileNotFoundError(
            f"DOC file not found: {path}"
        )

    word = None
    document = None

    with TemporaryDirectory() as temp_dir:
        converted_path = Path(temp_dir) / "converted.docx"

        try:
            word = win32com.client.DispatchEx("Word.Application")
            word.Visible = False
            word.DisplayAlerts = 0

            document = word.Documents.Open(
                str(path.resolve()),
                ReadOnly=True,
                AddToRecentFiles=False,
            )

            # 16 is Word's format code for DOCX.
            document.SaveAs2(
                str(converted_path),
                FileFormat=16,
            )

            document.Close(False)
            document = None

        finally:
            if document is not None:
                document.Close(False)

            if word is not None:
                word.Quit()

        return load_docx(converted_path)
