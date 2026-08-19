"""Read exact source text and validate dataset JSON input and output."""

from .modele import StudentSearchResults
from pathlib import Path
from .modele import StudentSearchResultsAndAnswer
from pydantic import ValidationError
import json

from typing import TypeVar

T = TypeVar("T")


def read_source_file(file_path: str | Path) -> str:
    """Read a UTF-8 file without translating its line endings.

    Args:
        file_path: Source file to read.

    Returns:
        Exact decoded text, including original line endings.

    Raises:
        OSError: If the file cannot be opened.
        UnicodeError: If the file is not valid UTF-8.
    """
    path = Path(file_path)
    with path.open(mode="r", encoding="utf-8", newline="") as file:
        return file.read()


def read_json_file(file_path: str, model: type[T]) -> T:
    """Read a JSON object into a model with descriptive validation errors.

    Args:
        file_path: UTF-8 JSON input path.
        model: Model class used to validate the JSON object.

    Returns:
        A validated instance of the requested model.

    Raises:
        FileNotFoundError: If the input path does not exist.
        OSError: If the input file cannot be opened or read.
        ValueError: If UTF-8, JSON syntax, or dataset structure is invalid.
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Invalid dataset structure: {file_path}")
        s = model(**data)
        return s
    except FileNotFoundError as error:
        raise FileNotFoundError(f"File not found: {file_path}") from error
    except UnicodeError as error:
        raise ValueError(f"File is not valid UTF-8: {file_path}") from error
    except OSError as error:
        raise OSError(f"Cannot read JSON file {file_path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Malformed JSON in {file_path} at line {error.lineno}, "
            f"column {error.colno}: {error.msg}"
        ) from error
    except ValidationError as error:
        raise ValueError(f"Invalid dataset structure in {file_path}: {error}")


def save_json(
    save_directory: str,
    output: StudentSearchResults | StudentSearchResultsAndAnswer,
    input_path: str,
) -> None:
    """Save validated results under the input filename in an output folder.

    Args:
        save_directory: Destination directory, created if necessary.
        output: Validated search results, optionally including answers.
        input_path: Input path whose basename determines the output name.

    Raises:
        OSError: If the destination cannot be created or written.
    """
    save = output.model_dump_json(indent=2)
    save_dir = Path(save_directory)
    output_path = save_dir / Path(input_path).name
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
        output_path.write_text(save, encoding="utf-8")
    except OSError as error:
        raise OSError(
            f"Cannot save JSON results to {output_path}: {error}"
        ) from error
