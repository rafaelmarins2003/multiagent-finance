"""Generate a deterministic portfolio/profile workload for evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from pydantic import TypeAdapter

from mafin.data.workload import generate_workload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None, help="Path to write workload JSON.")
    parser.add_argument("--portfolios", type=int, default=10, help="Number of base portfolios.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed.")
    parser.add_argument("--min-holdings", type=int, default=4)
    parser.add_argument("--max-holdings", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workload = generate_workload(
        portfolios=args.portfolios,
        seed=args.seed,
        min_holdings=args.min_holdings,
        max_holdings=args.max_holdings,
    )
    payload = TypeAdapter(dict).dump_json(workload, indent=2).decode("utf-8")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{payload}\n", encoding="utf-8")
        print(f"Wrote workload to {args.output}")
        return

    print(payload)


if __name__ == "__main__":
    main()
