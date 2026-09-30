from pathlib import Path


def dfs_file(path: Path) -> list[Path]:
    """Walk files deterministically, skipping symbolic links.

    Args:
        path: File or directory at which traversal begins.

    Returns:
        Supported source file paths in deterministic traversal order.
    """
    data: list[Path] = []
    if path.is_dir():
        for child in sorted(path.iterdir()):
            if child.is_symlink():
                continue
            if child.is_dir():
                data.extend(dfs_file(child))
            else:
                if is_useful_file(child):
                    data.append(child)
    else:
        if is_useful_file(path):
            data.append(path)

    return data


def is_useful_file(path: Path) -> bool:
    """Return whether the file has a supported text extension.

    Args:
        path: Candidate source file path.

    Returns:
        True for Python, Markdown, and plain-text extensions.
    """
    useful = [".py", ".txt", ".md"]
    return path.suffix in useful


def find_files(directory: str) -> list[Path]:
    """Collect supported files or reject a nonexistent input path.

    Args:
        directory: Source directory or individual file.

    Returns:
        Supported source file paths.

    Raises:
        ValueError: If the input path does not exist.
    """
    path = Path(directory).resolve()
    try:
        path = path.relative_to(Path.cwd().resolve())
    except ValueError as error:
        raise ValueError("Corpus must be inside the project") from error
    if not path.exists():
        raise ValueError(f"Corpus path does not exist: {directory}")
    return dfs_file(path)
