import os
import time
from collections import namedtuple

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


def load_model(modelpath):
    """Load an optional local Hugging Face model.

    API-only DeepSeek runs do not need torch/transformers. Keeping these imports
    local makes the default installation small and compatible with Python 3.13.
    """
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise RuntimeError(
            "Local models require optional packages: pip install torch transformers accelerate"
        ) from exc

    print(f"loading from {modelpath}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(modelpath)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        modelpath, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).cuda()
    return model, tokenizer


def load_via_deepspeed(model_name):
    try:
        import deepspeed
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError(
            "DeepSpeed loading requires optional packages: torch transformers deepspeed"
        ) from exc

    config = AutoConfig.from_pretrained(model_name)
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    dtype = config.torch_dtype
    hidden_size = config.hidden_size
    ds_config = {
        "fp16": {"enabled": dtype == torch.float16},
        "bf16": {"enabled": dtype == torch.bfloat16},
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": hidden_size * hidden_size,
            "stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
            "stage3_param_persistence_threshold": 0,
        },
        "steps_per_print": 2000,
        "train_batch_size": world_size,
        "train_micro_batch_size_per_gpu": 1,
        "wall_clock_breakdown": False,
    }
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).eval()
    engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
    engine.module.eval()
    return engine.module


def _as_request_result(texts):
    Completion = namedtuple("Completion", ["text"])
    RequestResult = namedtuple("RequestResult", ["completions", "success", "embedding", "cached"])
    return RequestResult(
        completions=[Completion(text=text) for text in texts],
        success=True,
        embedding=None,
        cached=False,
    )


def gen_from_prompt(
    model,
    tokenizer,
    prompt,
    echo_prompt=False,
    temperature=0.0,
    max_tokens=20,
    num_completions=1,
    output_scores=False,
    service=None,
    seed=101,
    process_func=None,
    terminate_by_linebreak=True,
    verbose=False,
    use_helm=False,
    auth=None,
):
    del output_scores
    if service is None:
        if model is None:
            raise ValueError("A local model or API service is required")
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Local inference requires torch") from exc
        if process_func is not None:
            prompt = process_func(prompt)
        prompt_ids = tokenizer(prompt, return_tensors="pt", padding=True)
        attention_mask = prompt_ids["attention_mask"].to(model.device)
        input_ids = prompt_ids["input_ids"].to(model.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                temperature=temperature,
                do_sample=temperature > 0,
                max_length=max_tokens + input_ids.size(1),
                num_return_sequences=num_completions,
                eos_token_id=2,
                pad_token_id=2,
            )
        texts = tokenizer.batch_decode(
            generated_ids[:, input_ids.size(1) :], skip_special_tokens=True
        )
        if terminate_by_linebreak != "no":
            texts = [text.split("\n")[0] for text in texts]
        return _as_request_result(texts)

    if model.startswith(("gpt", "deepseek")):
        texts = query_openai_compatible(
            client=service,
            model=model,
            prompt_lst=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            num_completions=num_completions,
            verbose=verbose,
        )
        return _as_request_result(texts)

    if model.startswith("claude"):
        texts = query_claude(
            client=service,
            model=model,
            prompt_lst=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            num_completions=num_completions,
            verbose=verbose,
        )
        return _as_request_result(texts)

    if use_helm:
        from helm.common.request import Request

        if len(prompt) != 1:
            raise ValueError("HELM supports one prompt per request in this project")
        for retry in range(5):
            try:
                request = Request(
                    model=model,
                    prompt=prompt[0],
                    echo_prompt=echo_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    num_completions=num_completions,
                    random=str(seed),
                    stop_sequences=["\n"],
                )
                return service.make_request(auth, request)
            except Exception:
                if retry == 4:
                    raise
                time.sleep(10)
    raise NotImplementedError(f"Unsupported model: {model}")


def query_claude(client, model, prompt_lst, temperature, max_tokens, num_completions, verbose, max_num_retries=5):
    results = []
    for prompt in prompt_lst:
        message = None
        for retry in range(max_num_retries):
            try:
                message = client.messages.create(
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                )
                break
            except Exception:
                if retry == max_num_retries - 1:
                    raise
                time.sleep(10)
        text = message.content[0].text.strip()
        if verbose:
            print(text)
        results.append(text)
    return results


def query_openai_compatible(
    client, model, prompt_lst, temperature, max_tokens, num_completions, verbose, max_num_retries=5
):
    results = []
    for prompt in prompt_lst:
        completion = None
        for retry in range(max_num_retries):
            try:
                request_kwargs = dict(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful AI agent."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=num_completions,
                )
                # V4 models can spend the entire small token budget on hidden
                # reasoning. AutoBencher expects the original non-thinking
                # ChatCompletions behavior, especially for 20-token judgments.
                if model.startswith("deepseek-v4"):
                    request_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
                completion = client.chat.completions.create(**request_kwargs)
                content = completion.choices[0].message.content
                if not content or not content.strip():
                    raise ValueError("API returned an empty completion")
                break
            except Exception as exc:
                if retry == max_num_retries - 1:
                    raise RuntimeError(
                        f"API request failed after {max_num_retries} attempts: {exc}"
                    ) from exc
                print(f"API request failed ({type(exc).__name__}); retrying...")
                time.sleep(10)
        text = content.strip()
        if verbose:
            print(text)
        results.append(text)
    return results


# Backward-compatible name used by older callers.
query_gpt4 = query_openai_compatible


def helm_process_args(experiment_model):
    try:
        from helm.common.authentication import Authentication
        from helm.proxy.services.remote_service import RemoteService
    except ImportError as exc:
        raise RuntimeError("HELM mode requires the optional crfm-helm package") from exc
    api_key = os.getenv("CRFM_API_KEY")
    if not api_key:
        raise RuntimeError("CRFM_API_KEY is not configured")
    auth = Authentication(api_key=api_key)
    service = RemoteService("https://crfm-models.stanford.edu")
    print(service.get_account(auth).usages)
    return experiment_model.lower(), None, service, auth


def process_args_for_models(experiment_model):
    if experiment_model.startswith("deepseek"):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not configured; add it to .env")
        model_client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
        return experiment_model, None, experiment_model, model_client

    if experiment_model.startswith("gpt"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        model_client = OpenAI(api_key=api_key, organization=os.getenv("OPENAI_ORG_ID"))
        return experiment_model, None, experiment_model, model_client

    if experiment_model.startswith("claude"):
        from anthropic import Anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not configured")
        return experiment_model, None, experiment_model, Anthropic(api_key=api_key)

    model, tokenizer = load_model(experiment_model)
    return model, tokenizer, os.path.basename(experiment_model).replace("/", "_"), None
