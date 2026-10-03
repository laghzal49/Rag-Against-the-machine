"""Fetch the supplied vLLM corpus for a fresh project checkout."""

import os
import subprocess
import tempfile
from pathlib import Path
from zipfile import BadZipFile, ZipFile

CORPUS_URL = (
    "https://cdn.intra.42.fr/document/document/54840/vllm-0.10.1.zip"
)
CORPUS_NAME = "vllm-0.10.1"


def ensure_default_corpus(data_path: str) -> None:
    """Download the default corpus if its Git-ignored directory is absent.

    Args:
        data_path: Corpus directory requested by the index command.

    Raises:
        OSError: If downloading or extracting the archive fails.
        ValueError: If the archive has unexpected paths or contents.
    """
    source_dir = Path(data_path)
    if source_dir != Path("data/raw"):
        return

    destination = source_dir / CORPUS_NAME
    if (destination / "README.md").is_file():
        return
    if destination.exists():
        raise ValueError(f"Incomplete corpus directory: {destination}")

    source_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".vllm-download-", dir=source_dir
    ) as temporary:
        work_dir = Path(temporary)
        archive = work_dir / "vllm-0.10.1.zip"
        try:
            subprocess.run(
                [
                    "wget", "-q", "--timeout=30", "--tries=2",
                    CORPUS_URL, "-O", str(archive),
                ],
                check=True,
                timeout=90,
            )
        except (
            FileNotFoundError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as error:
            raise OSError(
                f"Could not download the vLLM corpus from {CORPUS_URL}"
            ) from error

        extracted = work_dir / "extracted"
        try:
            with ZipFile(archive) as zipped:
                for member in zipped.namelist():
                    path = Path(member)
                    if path.is_absolute() or not path.parts or (
                        path.parts[0] != CORPUS_NAME
                        or ".." in path.parts
                    ):
                        raise ValueError(
                            "The vLLM archive contains an unexpected path"
                        )
                zipped.extractall(extracted)
        except BadZipFile as error:
            raise ValueError("The vLLM archive is not a valid ZIP") from error

        corpus = extracted / CORPUS_NAME
        if not (corpus / "README.md").is_file():
            raise ValueError("The vLLM archive is missing its README")
        os.replace(corpus, destination)
