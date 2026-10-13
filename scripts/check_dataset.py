"""Check whether a local MVTec AD category has the expected layout."""

from __future__ import annotations

import argparse
import json
import sys

from visionguard.config import load_config, resolve_project_path
from visionguard.data.validation import validate_category


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", help="MVTec AD root (defaults to project configuration).")
    parser.add_argument("--category", help="Category name (defaults to project configuration).")
    args = parser.parse_args()
    config = load_config()
    dataset_root = resolve_project_path(args.root or config["dataset"]["root"])
    category = args.category or config["dataset"]["category"]
    report = validate_category(dataset_root, category)
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        print(
            "\nPlace the extracted MVTec AD category under data/mvtec_ad and "
            "check its official usage conditions.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
