from pathlib import Path


def load_txt(file_path: str | Path) -> str:
    """Read a .txt file and return its contents as a string."""
    path = Path(file_path)

    if path.suffix.lower() != ".txt":
        raise ValueError(
            f"Expected a .txt file, got: {path.name}"
        )

    return path.read_text(
        encoding="utf-8-sig",
        errors="replace",
    )
