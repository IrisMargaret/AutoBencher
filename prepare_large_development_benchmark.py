#!/usr/bin/env python3
"""Create and freeze the 540-item static development stress set on VEPFS."""

import argparse
import json

from autobencher.large_dev_benchmark import write_large_development_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allowed-data-root", required=True)
    args = parser.parse_args()
    print(json.dumps(write_large_development_benchmark(args.output, args.allowed_data_root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
