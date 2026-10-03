"""Run the src Fire CLI with a concise top-level error boundary."""

import sys
import argparse
from fire import Fire
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
        if len(sys.argv) == 1 or sys.argv[1] in ("-h", "--help"):
            print(
                "Usage: python -m src "
                "{index,search,search_dataset,answer,answer_dataset,evaluate} "
                "[options]"
            )
            return 0
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
        Fire(
            selected,
            command=remaining,
        )
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except SystemExit as error:
        return error.code if isinstance(error.code, int) else 1
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except BaseException as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
