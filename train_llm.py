"""Built-in offline QLoRA fine-tuning for AutoBencher math datasets."""

import argparse
import importlib
import inspect
import json
import logging
import os
import re
import sys
import tempfile
from pathlib import Path


LOGGER = logging.getLogger("TrainLLM")
ALPACA_FIELDS = {"instruction", "input", "output"}
REQUIRED_PACKAGES = (
    "torch",
    "transformers",
    "peft",
    "bitsandbytes",
    "trl",
    "datasets",
    "accelerate",
)


# [ADDED] Use a stable, machine-readable log prefix.
def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[TrainLLM] %(levelname)s %(message)s",
        stream=sys.stdout,
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune a local Qwen-compatible causal language model with "
            "4-bit QLoRA and merge the resulting adapter."
        )
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Effective batch size implemented through gradient accumulation.",
    )
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    return parser


# [ADDED] Fail before model loading when an optional training package is absent.
def validate_dependencies():
    failures = []
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
        except Exception as exc:
            failures.append(f"{package} ({type(exc).__name__})")
    if failures:
        raise RuntimeError(
            "Missing or incompatible fine-tuning packages: "
            + ", ".join(failures)
            + ". Install the training dependency set from requirements-lock.txt."
        )


def _load_model_map():
    raw_mapping = os.getenv("AUTOBENCHER_MODEL_MAP", "").strip()
    if not raw_mapping:
        return {}
    mapping_path = Path(raw_mapping)
    if mapping_path.is_file():
        raw_mapping = mapping_path.read_text(encoding="utf-8")
    try:
        mapping = json.loads(raw_mapping)
    except json.JSONDecodeError as exc:
        raise ValueError("AUTOBENCHER_MODEL_MAP must be a JSON object or file") from exc
    if not isinstance(mapping, dict):
        raise ValueError("AUTOBENCHER_MODEL_MAP must contain a JSON object")
    return {str(key): str(value) for key, value in mapping.items()}


# [ADDED] Resolve an evaluation alias to local Hugging Face-compatible weights.
def resolve_model_source(model_name_or_path):
    requested = str(model_name_or_path).strip()
    direct_path = Path(requested).expanduser()
    if direct_path.is_dir():
        return str(direct_path.resolve())

    mapping = _load_model_map()
    mapped = mapping.get(requested)
    if mapped:
        mapped_path = Path(mapped).expanduser()
        if not mapped_path.is_dir():
            raise FileNotFoundError(
                f"Mapped local model directory does not exist: {mapped_path}"
            )
        return str(mapped_path.resolve())

    configured_path = os.getenv("AUTOBENCHER_LOCAL_MODEL_PATH", "").strip()
    if configured_path:
        local_path = Path(configured_path).expanduser()
        if local_path.is_dir():
            return str(local_path.resolve())

    local_root = os.getenv("AUTOBENCHER_LOCAL_MODEL_ROOT", "").strip()
    if local_root:
        root = Path(local_root).expanduser()
        aliases = {
            requested,
            requested.replace(":", "-"),
            requested.replace(":", "_"),
        }
        for alias in aliases:
            candidate = root / alias
            if candidate.is_dir():
                return str(candidate.resolve())

    drive, _ = os.path.splitdrive(requested)
    if ":" in requested and not drive:
        raise FileNotFoundError(
            "An Ollama tag cannot be fine-tuned directly. Set "
            "AUTOBENCHER_LOCAL_MODEL_PATH or map the tag to its original local "
            "Hugging Face model directory with AUTOBENCHER_MODEL_MAP."
        )
    # A repository-style identifier is accepted only from the local HF cache.
    return requested


# [ADDED] Accept only the exact Alpaca JSONL schema exported by AutoBencher.
def load_alpaca_records(dataset_path):
    path = Path(dataset_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Training dataset does not exist: {path}")
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL record at line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(record, dict) or set(record) != ALPACA_FIELDS:
                raise ValueError(
                    f"Line {line_number} must contain exactly: "
                    "instruction, input, output"
                )
            normalized = {
                key: str(record[key]).strip()
                for key in ("instruction", "input", "output")
            }
            if not all(normalized.values()):
                raise ValueError(
                    f"Line {line_number} contains an empty Alpaca field"
                )
            if any(
                re.search(r"[\u3400-\u9fff]", value)
                for value in normalized.values()
            ):
                raise ValueError(
                    f"Line {line_number} contains non-English CJK text"
                )
            records.append(normalized)
    if not records:
        raise ValueError("Training dataset contains no usable Alpaca records")
    return records


def _format_record(tokenizer, record):
    user_content = record["instruction"]
    if record["input"]:
        user_content += "\n\n" + record["input"]
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": record["output"]},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        text = (
            "### Instruction:\n"
            f"{record['instruction']}\n\n"
            "### Input:\n"
            f"{record['input']}\n\n"
            "### Response:\n"
            f"{record['output']}"
        )
    if tokenizer.eos_token and not text.endswith(tokenizer.eos_token):
        text += tokenizer.eos_token
    return text


def _supported_kwargs(callable_object, values):
    signature = inspect.signature(callable_object)
    return {
        key: value
        for key, value in values.items()
        if key in signature.parameters
    }


def _training_config(args, adapter_output, use_bfloat16):
    from transformers import TrainingArguments

    values = {
        "output_dir": str(adapter_output),
        "num_train_epochs": args.epoch,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": args.batch,
        "learning_rate": args.learning_rate,
        "logging_steps": 1,
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "optim": "paged_adamw_8bit",
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "bf16": use_bfloat16,
        "fp16": not use_bfloat16,
        "gradient_checkpointing": True,
        "report_to": "none",
        "remove_unused_columns": False,
        "dataloader_pin_memory": True,
        "seed": 42,
        "data_seed": 42,
    }
    try:
        from trl import SFTConfig

        sft_values = dict(values)
        sft_values.update(
            {
                "dataset_text_field": "text",
                "max_length": args.max_seq_length,
                "max_seq_length": args.max_seq_length,
                "packing": False,
            }
        )
        return SFTConfig(
            **_supported_kwargs(SFTConfig.__init__, sft_values)
        )
    except ImportError:
        return TrainingArguments(
            **_supported_kwargs(TrainingArguments.__init__, values)
        )


# [ADDED] Train a QLoRA adapter, merge it, and save a complete local model.
def train_and_merge(args, model_source, records):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for 4-bit bitsandbytes QLoRA")
    use_bfloat16 = bool(
        hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )
    compute_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
    LOGGER.info("stage=tokenizer_load source=%s", model_source)
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    formatted_records = [
        {"text": _format_record(tokenizer, record)}
        for record in records
    ]
    dataset = Dataset.from_list(formatted_records)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    LOGGER.info("stage=model_load quantization=4bit source=%s", model_source)
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        local_files_only=True,
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map={"": 0},
        torch_dtype=compute_dtype,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"{output_path.name}.adapter.",
        dir=output_path.parent,
    ) as adapter_directory:
        adapter_output = Path(adapter_directory)
        training_args = _training_config(
            args,
            adapter_output,
            use_bfloat16,
        )
        trainer_values = {
            "model": model,
            "args": training_args,
            "train_dataset": dataset,
            "processing_class": tokenizer,
            "tokenizer": tokenizer,
            "dataset_text_field": "text",
            "max_seq_length": args.max_seq_length,
            "packing": False,
            "peft_config": lora_config,
        }
        trainer = SFTTrainer(
            **_supported_kwargs(SFTTrainer.__init__, trainer_values)
        )
        LOGGER.info(
            "stage=train samples=%d epochs=%d effective_batch=%d lora_rank=%d",
            len(records),
            args.epoch,
            args.batch,
            args.lora_rank,
        )
        trainer.train()
        trainer.model.save_pretrained(
            adapter_output,
            safe_serialization=True,
        )
        del trainer
        del model
        torch.cuda.empty_cache()

        LOGGER.info("stage=merge adapter=%s", adapter_output)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_source,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=compute_dtype,
            low_cpu_mem_usage=True,
            device_map={"": "cpu"},
        )
        adapter_model = PeftModel.from_pretrained(
            base_model,
            adapter_output,
            local_files_only=True,
        )
        merged_model = adapter_model.merge_and_unload()
        output_path.mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(
            output_path,
            safe_serialization=True,
            max_shard_size="4GB",
        )
        tokenizer.save_pretrained(output_path)
        generation_config = getattr(merged_model, "generation_config", None)
        if generation_config is not None:
            generation_config.save_pretrained(output_path)
        del adapter_model
        del merged_model
        del base_model
        torch.cuda.empty_cache()
    LOGGER.info("stage=complete output=%s", output_path)


def _is_complete_model_directory(output_path):
    path = Path(output_path).expanduser()
    if not (path / "config.json").is_file():
        return False
    if not (
        (path / "tokenizer_config.json").is_file()
        or (path / "tokenizer.json").is_file()
    ):
        return False
    weight_patterns = (
        "model*.safetensors",
        "pytorch_model*.bin",
    )
    return any(list(path.glob(pattern)) for pattern in weight_patterns)


def main(argv=None):
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("epoch", "batch", "lora_rank", "max_seq_length"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be at least 1")
    if args.learning_rate <= 0:
        parser.error("--learning_rate must be greater than zero")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if _is_complete_model_directory(args.output_path):
        LOGGER.info("stage=resume output_already_complete=%s", args.output_path)
        return 0
    try:
        validate_dependencies()
    except RuntimeError as exc:
        LOGGER.error("stage=dependency_check error=%s", exc)
        return 3
    try:
        records = load_alpaca_records(args.dataset_path)
        model_source = resolve_model_source(args.model_name_or_path)
    except (OSError, ValueError) as exc:
        LOGGER.error("stage=input_validation error=%s", exc)
        return 4
    try:
        LOGGER.info(
            "stage=start model=%s dataset=%s gpu=%s",
            model_source,
            Path(args.dataset_path).resolve(),
            args.gpu,
        )
        train_and_merge(args, model_source, records)
    except Exception as exc:
        error_text = re.sub(r"\s+", " ", str(exc)).strip()
        if "out of memory" in error_text.lower():
            LOGGER.error("stage=train failure=out_of_memory error=%s", error_text)
            return 5
        LOGGER.error(
            "stage=train failure=%s error=%s",
            type(exc).__name__,
            error_text,
        )
        if os.getenv("AUTOBENCHER_DEBUG_TRAINING", "").lower() in {
            "1",
            "true",
            "yes",
        }:
            LOGGER.exception("stage=train debug_traceback")
        return 6
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
