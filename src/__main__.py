"""Run the src Fire CLI with a concise top-level error boundary."""

import sys

import fire
from .commands import (
    answer,
    answer_dataset,
    index,
    search,
    evaluate,
    search_dataset,
)


def main() -> int:
    """Dispatch a command and return a shell-compatible exit status.

    Returns:
        Zero on success, one on failure, or 130 on interruption.
    """
    try:
        fire.Fire(
            {
                "search": search,
                "index": index,
                "answer": answer,
                "search_dataset": search_dataset,
                "answer_dataset": answer_dataset,
                "evaluate": evaluate,
            }
        )
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
