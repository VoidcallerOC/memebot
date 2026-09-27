"""CLI: python -m bot.meta

Read-only scan. Never talks to the executor.
"""
from __future__ import annotations

import json
import sys

from ..config import load_config
from .detector import MetaDetector
from .report import format_report


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cfg = load_config()
    detector = MetaDetector(cfg)
    report = detector.scan_live()
    if "--json" in argv:
        print(json.dumps({
            "generated_at": report.generated_at,
            "notes": report.notes,
            "clusters": [c.to_dict() for c in report.clusters],
            "signals": [s.to_dict() for s in report.signals],
        }, indent=2))
    else:
        print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
