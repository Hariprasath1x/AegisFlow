"""Champion/challenger operations. Separate from training on purpose: no training
run can make itself champion (ADR 0005 D8).

python scripts/champion.py --compare
python scripts/champion.py --promote --actor p.raman
python scripts/champion.py --rollback --actor p.raman
"""
from __future__ import annotations

import argparse
import sys

from aegis_ml.training.registry import compare, promote_challenger, rollback


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Champion/challenger operations for AegisFlow fraud models.",
    )
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--compare",
        action="store_true",
        help="Compare current champion and challenger without mutating aliases.",
    )
    group.add_argument(
        "--promote",
        action="store_true",
        help="Promote challenger to champion after re-validating gates and lift.",
    )
    group.add_argument(
        "--rollback",
        action="store_true",
        help="Roll back champion to the previous champion alias.",
    )
    ap.add_argument(
        "--actor",
        default=None,
        help="Human operator identity performing the mutating operation (required for --promote and --rollback).",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    actor = args.actor.strip() if args.actor else ""

    if args.promote or args.rollback:
        if not actor or actor.lower() == "unknown":
            print(
                "Error: --actor must be explicitly specified (and cannot be 'unknown') "
                "for mutating operations (--promote, --rollback).",
                file=sys.stderr,
            )
            return 2

    if args.rollback:
        d = rollback(actor=actor)
    elif args.promote:
        d = promote_challenger(actor=actor)
    elif args.compare:
        d = compare()
    else:
        parser.print_help(sys.stderr)
        return 2

    print(d.reason)
    return 0 if d.changed else 1


if __name__ == "__main__":
    sys.exit(main())
