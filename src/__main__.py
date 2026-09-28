"""Run the src Fire CLI with a concise top-level error boundary."""

import sys

import fire

from .cli import Cli


def main() -> int:
    """Dispatch a command and return a shell-compatible exit status.

    Returns:
        Zero on success, one on failure, or 130 on interruption.
    """
    cli = Cli()
    try:
        fire.Fire(
            {
                "search": cli.search,
                "index": cli.index,
                "answer": cli.answer,
                "search_dataset": cli.search_dataset,
                "answer_dataset": cli.answer_dataset,
                "evaluate": cli.evaluate,
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
