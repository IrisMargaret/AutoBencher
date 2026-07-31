"""Install the project-native fixed math holdout into the data filesystem."""

from __future__ import annotations

import argparse
import json

from autobencher.fixed_benchmark import install_project_fixed_test_set


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Install AutoBencher's project-native fixed math test set. "
            "This command performs no network or Hugging Face dataset access."
        )
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--allowed-data-root", required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace a different existing benchmark intentionally.",
    )
    args = parser.parse_args()
    result = install_project_fixed_test_set(
        args.output,
        allowed_data_root=args.allowed_data_root,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
