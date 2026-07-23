"""Low-cost end-to-end DeepSeek API and project import check."""

import importlib
import os

from util import gen_from_prompt, process_args_for_models


def main():
    from tool_util import extract_json_v2

    parsed = extract_json_v2(
        '```json\n[{"id": "1", "answer": "2"}]\n```', outfilename=None
    )
    if parsed != [[{"id": "1", "answer": "2"}]]:
        raise RuntimeError(f"JSON extraction failed: {parsed!r}")

    model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    model, tokenizer, _, client = process_args_for_models(model_name)
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

    for module in ("tool_util", "wiki_autobencher", "multilingual_autobencher", "math_autobencher"):
        importlib.import_module(module)
    print(f"Smoke test passed with {model_name}; all benchmark modules import successfully.")


if __name__ == "__main__":
    main()
