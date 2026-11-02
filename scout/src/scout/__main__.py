"""`python -m scout` is the `scout` command."""

import sys

from scout.surfaces.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
