"""Drobné pomôcky pre výpis do konzoly."""

from __future__ import annotations

import sys

LINE_WIDTH: int = 72


def configure_console() -> None:
    """Prepne výstup na UTF-8, aby diakritika neskončila na cp1252 chybe (Windows)."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def header(title: str) -> None:
    """Vypíše nadpis sekcie oddelený čiarami."""
    print()
    print("=" * LINE_WIDTH)
    print(title)
    print("=" * LINE_WIDTH)


def human_size(num_bytes: float) -> str:
    """Naformátuje veľkosť v bajtoch na čitateľný reťazec."""
    for unit in ("B", "kB", "MB", "GB"):
        if abs(num_bytes) < 1024.0 or unit == "GB":
            return f"{num_bytes:.1f} {unit}" if unit != "B" else f"{int(num_bytes)} B"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} GB"
