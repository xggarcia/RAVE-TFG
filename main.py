# -*- coding: utf-8 -*-
"""
RAVE-TFG main entrypoint.

This file intentionally stays small and delegates behavior to CLI modules.
"""

import sys


# Configure UTF-8 encoding for Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        import codecs

        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")


def main():
    """Program entrypoint for interactive and non-interactive execution."""
    if len(sys.argv) == 1:
        from src.cli.interactive_menu import interactive_menu

        interactive_menu()
        return 0

    from src.cli.commands import build_parser, run_command

    parser = build_parser()
    args = parser.parse_args()
    run_command(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
