"""CLI: python -m bot.meta [report|snapshot|run]

Read-only. Never talks to the executor.
"""
from __future__ import annotations

import json
import sys

from ..config import load_config
from .detector import MetaDetector
from .pipeline import MetaPipeline
from .report import format_report


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cfg = load_config()
    command = argv[0] if argv and not argv[0].startswith("-") else "report"
    flags = [a for a in argv if a.startswith("-")]
    if command == "snapshot":
        report = MetaPipeline(cfg).snapshot_universe()
    elif command == "run":
        MetaPipeline(cfg).run_forever()
        return 0
    else:
        report = MetaDetector(cfg).scan_live()
    if "--json" in flags:
        print(json.dumps({
            "generated_at": report.generated_at,
            "notes": report.notes,
            "window_coverage": report.window_coverage,
            "clusters": [c.to_dict() for c in report.clusters],
            "signals": [s.to_dict() for s in report.signals],
        }, indent=2))
    else:
        print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
