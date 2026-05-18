"""RAVE-TFG main entrypoint.

Without arguments, launches the PySide6 desktop app.
With arguments, dispatches to the argparse CLI in src/cli/commands.py.
"""

import sys


def main():
    if len(sys.argv) == 1:
        from app.__main__ import main as app_main

        return app_main()

    from src.cli.commands import build_parser, run_command

    parser = build_parser()
    args = parser.parse_args()
    run_command(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
