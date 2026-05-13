#!/usr/bin/env python3
from __future__ import annotations

import sys

# Compatibility layer: keep `python upmix_cli.py` and historical imports working.
from upmixer import *  # noqa: F401,F403
from upmixer.cli import main


if __name__ == "__main__":
    if "--gui" in sys.argv:
        from upmix_gui import main as gui_main

        gui_main()
    else:
        main()
