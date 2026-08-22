"""CLI: python -m mps_sentry"""

from __future__ import annotations

import argparse
import platform
import sys

from . import __version__, mps_usable, run
from .probes import ALL_PROBES

_ICON = {"ok": "  ok  ", "corrupt": "CORRUPT", "error": " error", "skipped": " skip "}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mps-sentry",
        description="Differential correctness checks for PyTorch's Apple Silicon (MPS) backend. "
        "Each check compares MPS against a CPU reference and flags disagreement that "
        "float rounding cannot explain.",
    )
    parser.add_argument("--only", nargs="+", choices=sorted(ALL_PROBES), help="run only these checks")
    parser.add_argument("--list", action="store_true", help="list available checks and exit")
    parser.add_argument("--version", action="version", version=f"mps-sentry {__version__}")
    args = parser.parse_args(argv)

    if args.list:
        for key, fn in sorted(ALL_PROBES.items()):
            summary = (fn.__doc__ or "").strip().splitlines()[0]
            print(f"  {key:20s} {summary}")
        return 0

    import torch

    print(f"mps-sentry {__version__}")
    print(f"torch {torch.__version__} | macOS {platform.mac_ver()[0]} | {platform.machine()}\n")

    usable, reason = mps_usable()
    if not usable:
        print(f"Cannot run: {reason}")
        print(
            "\nThese checks need a real Metal device. Virtualised macOS environments — including\n"
            "GitHub's macOS runners — report MPS as available but fail on the first allocation."
        )
        return 2

    results = run(args.only)
    width = max(len(r.name) for r in results)
    for r in results:
        diff = f"  max|Δ|={r.max_abs_diff:.3e}" if r.max_abs_diff is not None else ""
        print(f"[{_ICON[r.status]}] {r.name:<{width}}{diff}")
        if r.detail:
            print(f"           {r.detail}")
        if r.affected:
            print(f"           affected: {r.affected}")

    corrupt = [r for r in results if r.failed]
    print()
    if corrupt:
        print(f"{len(corrupt)} silent-correctness failure(s) on this torch build:")
        for r in corrupt:
            print(f"  - {r.name}")
        print("\nThese produce wrong numbers with no exception and no warning.")
        print("Upgrading torch is usually the fix; see the README for the known-affected versions.")
        return 1
    print("No silent-correctness failures detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
