import argparse
import os
from pathlib import Path
import subprocess
import sys


# [MODIFIED] Emit stable iteration metrics for eval and flywheel runs.
def log_math_iteration_metrics(
    iteration,
    global_accuracy,
    subcategory_coverage,
    hard_sample_count,
    directed_generation=False,
    triggers=None,
    weakest_sub_categories=None,
):
    trigger_text = ",".join(triggers or []) or "baseline"
    print(
        "[MathFlywheel] "
        f"iteration={iteration} "
        f"global_accuracy={global_accuracy:.3f} "
        f"subcategory_coverage={subcategory_coverage:.1%} "
        f"hard_samples={hard_sample_count} "
        f"directed_generation={'on' if directed_generation else 'off'} "
        f"trigger={trigger_text}"
    )
    if weakest_sub_categories:
        print("[MathFlywheel] top_10_weakest_sub_categories:")
        for item in weakest_sub_categories[:10]:
            print(
                "  "
                f"{item['category']} / {item['sub_category']}: "
                f"accuracy={item['accuracy']:.3f}"
            )


def _append_option(command, name, value):
    if value is not None:
        command.extend([name, str(value)])


def _option_present(argv, *names):
    return any(
        token == name or token.startswith(name + "=")
        for token in argv
        for name in names
    )


def _strip_implicit_config_options(command, argv):
    """Let YAML remain authoritative for launcher defaults not typed by users."""
    controlled = {
        "--agent_modelname": ("--agent_modelname", "--agent-modelname"),
        "--test_taker_modelname": (
            "--test_taker_modelname",
            "--test-taker-modelname",
        ),
        "--exp_mode": ("--exp_mode", "--exp-mode"),
        "--num_iters": ("--num_iters", "--num-iters"),
        "--outfile_prefix1": ("--outfile_prefix1", "--outfile-prefix1"),
        "--acc_target": ("--acc_target", "--acc-target"),
        "--mode": ("--mode",),
        "--export_interval": ("--export_interval", "--export-interval"),
        "--max_cycle": ("--max_cycle", "--max-cycle"),
        "--finetune_gpu": ("--finetune_gpu", "--finetune-gpu"),
        "--finetune_epoch": ("--finetune_epoch", "--finetune-epoch"),
        "--finetune_batch": ("--finetune_batch", "--finetune-batch"),
        "--lora_rank": ("--lora_rank", "--lora-rank"),
        "--new_local_model_suffix": (
            "--new_local_model_suffix",
            "--new-local-model-suffix",
        ),
        "--disk_warning_threshold": (
            "--disk_warning_threshold",
            "--disk-warning-threshold",
        ),
        "--clean_cycle_cache": (
            "--clean_cycle_cache",
            "--clean-cycle-cache",
        ),
    }
    stripped = []
    index = 0
    while index < len(command):
        token = command[index]
        aliases = controlled.get(token)
        if aliases and not _option_present(argv, *aliases):
            index += 2
            continue
        stripped.append(token)
        index += 1
    return stripped


# [ADDED] Parse explicit Boolean CLI values consistently.
def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        "Expected one of: true, false, yes, no, 1, 0, on, off"
    )


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
    execution_mode="eval",
    export_interval=1,
    max_cycle=1,
    finetune_gpu="0",
    finetune_epoch=3,
    finetune_batch=8,
    lora_rank=8,
    new_local_model_suffix="finetuned",
    disk_warning_threshold=10,
    clean_cycle_cache=True,
    config=None,
    run_id=None,
    resume=None,
    overrides=None,
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
    # [ADDED] Flywheel options are isolated to the math module.
    math_options = [
        "--mode", execution_mode,
        "--export_interval", str(export_interval),
        "--max_cycle", str(max_cycle),
        "--finetune_gpu", str(finetune_gpu),
        "--finetune_epoch", str(finetune_epoch),
        "--finetune_batch", str(finetune_batch),
        "--lora_rank", str(lora_rank),
        "--new_local_model_suffix", str(new_local_model_suffix),
        "--disk_warning_threshold", str(disk_warning_threshold),
        "--clean_cycle_cache", str(bool(clean_cycle_cache)).lower(),
    ]
    _append_option(math_options, "--config", config)
    _append_option(math_options, "--run_id", run_id)
    if resume is not None:
        _append_option(math_options, "--resume", str(bool(resume)).lower())
    if overrides:
        math_options.append("--override")
        math_options.extend(str(item) for item in overrides)
    return [
        sys.executable, "math_autobencher.py", *common, *math_options,
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
    # [ADDED] Built-in math data-flywheel controls.
    parser.add_argument(
        "--mode",
        dest="execution_mode",
        choices=["eval", "data_flywheel"],
        default="eval",
    )
    parser.add_argument(
        "--export-interval",
        "--export_interval",
        dest="export_interval",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max-cycle",
        "--max_cycle",
        dest="max_cycle",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--finetune-gpu",
        "--finetune_gpu",
        dest="finetune_gpu",
        default="0",
    )
    parser.add_argument(
        "--finetune-epoch",
        "--finetune_epoch",
        dest="finetune_epoch",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--finetune-batch",
        "--finetune_batch",
        dest="finetune_batch",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--lora-rank",
        "--lora_rank",
        dest="lora_rank",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--new-local-model-suffix",
        "--new_local_model_suffix",
        dest="new_local_model_suffix",
        default="finetuned",
    )
    parser.add_argument(
        "--disk-warning-threshold",
        "--disk_warning_threshold",
        dest="disk_warning_threshold",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--clean-cycle-cache",
        "--clean_cycle_cache",
        dest="clean_cycle_cache",
        type=parse_bool,
        default=True,
    )
    parser.add_argument("--config")
    parser.add_argument("--run-id", "--run_id", dest="run_id")
    parser.add_argument("--resume", type=parse_bool)
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        metavar="PATH=VALUE",
    )
    args = parser.parse_args()
    if args.mode != "math" and args.execution_mode != "eval":
        parser.error("--mode data_flywheel is supported only for math")
    for name in (
        "num_iters",
        "export_interval",
        "max_cycle",
        "finetune_epoch",
        "finetune_batch",
        "lora_rank",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be at least 1")
    if args.disk_warning_threshold < 0:
        parser.error("--disk_warning_threshold cannot be negative")
    # [MODIFIED] Announce the selected math execution mode and core metrics.
    if args.mode == "math":
        print(
            f"[MathFlywheel] mode={args.execution_mode}; metrics: "
            "global_accuracy, subcategory_coverage, hard_samples"
        )
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
        execution_mode=args.execution_mode,
        export_interval=args.export_interval,
        max_cycle=args.max_cycle,
        finetune_gpu=args.finetune_gpu,
        finetune_epoch=args.finetune_epoch,
        finetune_batch=args.finetune_batch,
        lora_rank=args.lora_rank,
        new_local_model_suffix=args.new_local_model_suffix,
        disk_warning_threshold=args.disk_warning_threshold,
        clean_cycle_cache=args.clean_cycle_cache,
        config=args.config,
        run_id=args.run_id,
        resume=args.resume,
        overrides=args.override,
    )
    if args.mode == "math" and args.config:
        command = _strip_implicit_config_options(command, sys.argv[1:])
    print("Running:", subprocess.list2cmdline(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
