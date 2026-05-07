#!/usr/bin/env python3
"""Backward-compatible wrapper for the packaged CLI."""

from __future__ import annotations

import sys

from amygda.cli import main

if __name__ == "__main__":
    main(["run", *sys.argv[1:]])
