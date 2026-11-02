"""Process exit and the error report of the CLI (design section 4): JSON error body on stdout, one human line
on stderr, exit code from the table."""

import sys

from scout.core.contracts.errors import ScoutError, cli_exit
from scout.surfaces.ops import error_hint

EXIT_OK = 0
EXIT_INTERRUPTED = 130


def report_error(err: ScoutError) -> int:
    print(err.to_body().model_dump_json())
    print(f"scout: error {err.code}: {err.message}{error_hint(err)}", file=sys.stderr)
    return cli_exit(err.code)
