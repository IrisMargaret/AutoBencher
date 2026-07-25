import argparse
import os
from pathlib import Path
import subprocess
import sys


def _append_option(command, name, value):
    if value is not None:
        command.extend([name, str(value)])


def build_command(
    mode,
    model,
    num_iters,
    *,
    agent_modelname=None,
    test_taker_modelname=None,
    test_taker_modelname2=None,
    tool_modelname=None,
    exp_mode="autobencher",
    use_helm="no",
    outfile_prefix1=None,
    acc_target=None,
    temperature=None,
    pairwise=None,
    theme=None,
    top_p=None,
):
    agent_modelname = agent_modelname or model
    test_taker_modelname = test_taker_modelname or model
    default_acc_target = "0.1--0.3" if mode in {"wiki", "math"} else "0.3--0.5"
    acc_target = acc_target or default_acc_target

    if outfile_prefix1 is None:
        if mode == "wiki":
            outfile_prefix1 = f"KI/history.{model}.0.1--0.3."
        elif mode == "multilingual":
            outfile_prefix1 = f"multilingual/5word_v3_{model}."
        else:
            outfile_prefix1 = f"math_v5/{model}.0.1--0.3."

    output_parent = Path(outfile_prefix1).parent
    if output_parent != Path("."):
        output_parent.mkdir(parents=True, exist_ok=True)

    common = [
        "--exp_mode", exp_mode,
        "--agent_modelname", agent_modelname,
        "--test_taker_modelname", test_taker_modelname,
        "--use_helm", use_helm,
        "--num_iters", str(num_iters),
        "--outfile_prefix1", outfile_prefix1,
        "--acc_target", acc_target,
    ]
    _append_option(common, "--test_taker_modelname2", test_taker_modelname2)
    _append_option(common, "--tool_modelname", tool_modelname)
    _append_option(common, "--temperature", temperature)
    _append_option(common, "--pairwise", pairwise)
    _append_option(common, "--top_p", top_p)

    if mode == "wiki":
        _append_option(common, "--theme", theme or "history")
        return [
            sys.executable, "wiki_autobencher.py", *common,
        ]
    if mode == "multilingual":
        return [
            sys.executable, "multilingual_autobencher.py", *common,
        ]
    return [
        sys.executable, "math_autobencher.py", *common,
    ]


def main():
    parser = argparse.ArgumentParser(description="Run an AutoBencher benchmark")
    parser.add_argument("mode", choices=["wiki", "multilingual", "math"])
    parser.add_argument(
        "--model",
        help="shorthand that sets both the agent and test-taker model",
    )
    parser.add_argument("--agent-modelname", "--agent_modelname", dest="agent_modelname")
    parser.add_argument(
        "--test-taker-modelname",
        "--test_taker_modelname",
        dest="test_taker_modelname",
    )
    parser.add_argument(
        "--test-taker-modelname2",
        "--test_taker_modelname2",
        dest="test_taker_modelname2",
    )
    parser.add_argument("--tool-modelname", "--tool_modelname", dest="tool_modelname")
    parser.add_argument("--exp-mode", "--exp_mode", dest="exp_mode", default="autobencher")
    parser.add_argument("--use-helm", "--use_helm", dest="use_helm", default="no")
    parser.add_argument("--num-iters", "--num_iters", dest="num_iters", type=int, default=8)
    parser.add_argument("--outfile-prefix1", "--outfile_prefix1", dest="outfile_prefix1")
    parser.add_argument("--acc-target", "--acc_target", dest="acc_target")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--pairwise")
    parser.add_argument("--theme")
    parser.add_argument("--top-p", "--top_p", dest="top_p", type=float)
    args = parser.parse_args()
    model = args.model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    command = build_command(
        args.mode,
        model,
        args.num_iters,
        agent_modelname=args.agent_modelname,
        test_taker_modelname=args.test_taker_modelname,
        test_taker_modelname2=args.test_taker_modelname2,
        tool_modelname=args.tool_modelname,
        exp_mode=args.exp_mode,
        use_helm=args.use_helm,
        outfile_prefix1=args.outfile_prefix1,
        acc_target=args.acc_target,
        temperature=args.temperature,
        pairwise=args.pairwise,
        theme=args.theme,
        top_p=args.top_p,
    )
    print("Running:", subprocess.list2cmdline(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
