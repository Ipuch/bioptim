"""
Child-process entry point for the codegen builder.

Invoked as ``python -m bioptim.models.codegen <spec.json>``. It exists as its own module so
that running it does not re-enter ``builder``, which the package ``__init__`` has already
imported -- ``python -m bioptim.models.codegen.builder`` would warn about exactly that.
"""

import sys

from .builder import _main

if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1]))
