import argparse
import os
from pathlib import Path
import subprocess
import sys


def build_command(mode, model, num_iters):
    common = [
        "--exp_mode", "autobencher",
        "--agent_modelname", model,
        "--test_taker_modelname", model,
        "--use_helm", "no",
        "--num_iters", str(num_iters),
    ]
    if mode == "wiki":
        Path("KI").mkdir(exist_ok=True)
        return [
            sys.executable, "wiki_autobencher.py", *common,
            "--theme", "history",
            "--outfile_prefix1", f"KI/history.{model}.0.1--0.3.",
            "--tool_modelname", model,
            "--acc_target", "0.1--0.3",
        ]
    if mode == "multilingual":
        Path("multilingual").mkdir(exist_ok=True)
        return [
            sys.executable, "multilingual_autobencher.py", *common,
            "--outfile_prefix1", f"multilingual/5word_v3_{model}.",
        ]
    Path("math_v5").mkdir(exist_ok=True)
    return [
        sys.executable, "math_autobencher.py", *common,
        "--outfile_prefix1", f"math_v5/{model}.0.1--0.3.",
        "--acc_target", "0.1--0.3",
    ]


def main():
    parser = argparse.ArgumentParser(description="Run an AutoBencher benchmark")
    parser.add_argument("mode", choices=["wiki", "multilingual", "math"])
    parser.add_argument(
        "--model",
        default=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        help="API or local model name",
    )
    parser.add_argument("--num-iters", type=int, default=8)
    args = parser.parse_args()
    command = build_command(args.mode, args.model, args.num_iters)
    print("Running:", subprocess.list2cmdline(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
