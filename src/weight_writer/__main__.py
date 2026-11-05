"""Allow ``python -m weight_writer`` to invoke the command-line interface."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
