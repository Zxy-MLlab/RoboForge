"""Deprecated source-checkout compatibility entry point.

This module is intentionally not installed as a console script. New runs use
``roboforge``/``roboforge-openhands``; it remains executable only so historical
run manifests and the legacy regression suite stay reproducible.
"""

from .cli import main

raise SystemExit(main())
