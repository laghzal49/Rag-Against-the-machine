"""Run the src Fire CLI with a concise top-level error boundary."""

import sys
import argparse
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
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            "command",
            choices=[
                "search",
                "index",
                "answer",
                "search_dataset",
                "answer_dataset",
                "evaluate",
            ],
        )

        args, remaining = parser.parse_known_args()
        commands = {
            "search": search,
            "index": index,
            "answer": answer,
            "search_dataset": search_dataset,
            "answer_dataset": answer_dataset,
            "evaluate": evaluate,
        }
        selected = commands[args.command]
        fire.Fire(
            selected,
            command=remaining,
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
