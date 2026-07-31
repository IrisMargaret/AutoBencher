"""Offline smoke check with an explicitly enabled optional API probe."""

import argparse
import os

from util import gen_from_prompt, process_args_for_models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run offline import/schema checks; use --api to add one paid "
            "model-provider request."
        )
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="also send one real API request (disabled by default)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        help="provider model used only with --api",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    from tool_util import extract_json_v2

    parsed = extract_json_v2(
        '```json\n[{"id": "1", "answer": "2"}]\n```', outfilename=None
    )
    if parsed != [[{"id": "1", "answer": "2"}]]:
        raise RuntimeError(f"JSON extraction failed: {parsed!r}")

    import math_autobencher

    del math_autobencher
    if args.api:
        model, tokenizer, _, client = process_args_for_models(args.model)
        result = gen_from_prompt(
            model=model,
            tokenizer=tokenizer,
            prompt=["Reply with exactly: AUTOBENCHER_OK"],
            service=client,
            temperature=0,
            max_tokens=20,
        )
        response = result.completions[0].text
        if "AUTOBENCHER_OK" not in response:
            raise RuntimeError(f"Unexpected API response: {response!r}")
        print(f"Smoke test passed with API model {args.model}.")
    else:
        print("Offline smoke test passed; no API request was sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
