import os
import time
from collections import namedtuple

import requests
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


def _ollama_model_name(model_name):
    """Return the Ollama model id when the CLI value uses Ollama syntax."""
    if model_name.startswith("ollama/"):
        return model_name.removeprefix("ollama/")

    drive, _ = os.path.splitdrive(model_name)
    if ":" in model_name and not drive:
        return model_name
    return None


def _ollama_base_url():
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    return base_url


class OllamaService:
    """Small native Ollama client with explicit preload and retry support."""

    def __init__(self, base_url=None, api_key=None):
        self.base_url = (base_url or _ollama_base_url()).rstrip("/")
        self.keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
        self.max_retries = int(os.getenv("OLLAMA_MAX_RETRIES", "10"))
        self.retry_delay = float(os.getenv("OLLAMA_RETRY_DELAY_SECONDS", "5"))
        self.request_timeout = float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "300"))
        self.session = requests.Session()
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"
        self.loaded_models = set()


def _ollama_error_detail(response):
    try:
        payload = response.json()
        return payload.get("error") or response.text
    except (ValueError, AttributeError):
        return getattr(response, "text", "")


def _request_ollama_json(service, path, payload):
    last_error = None
    for attempt in range(1, service.max_retries + 1):
        response = None
        try:
            response = service.session.post(
                f"{service.base_url}{path}",
                json=payload,
                timeout=(5, service.request_timeout),
            )
            response.raise_for_status()
            result = response.json()
            if result.get("error"):
                raise RuntimeError(result["error"])
            return result
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            detail = _ollama_error_detail(exc.response)
            last_error = f"HTTP {status}: {detail or exc}"
            if status is not None and 400 <= status < 500 and status not in {408, 429}:
                raise RuntimeError(
                    f"Ollama rejected the request for model {payload.get('model')!r}: "
                    f"{last_error}. Check that the model is installed."
                ) from exc
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_error = str(exc)

        if attempt < service.max_retries:
            print(
                f"Ollama request failed ({attempt}/{service.max_retries}): "
                f"{last_error}; retrying in {service.retry_delay:g}s..."
            )
            time.sleep(service.retry_delay)

    raise RuntimeError(
        f"Ollama request failed after {service.max_retries} attempts: {last_error}. "
        f"Verify that Ollama is running at {service.base_url} and has enough free memory."
    )


def query_ollama(
    service,
    model,
    prompt_lst,
    temperature,
    max_tokens,
    num_completions,
    verbose,
):
    # Existing callers only consume the first completion. Preserve that
    # behavior while using Ollama's native endpoint.
    del num_completions

    if model not in service.loaded_models:
        print(f"Preloading Ollama model {model}...")
        _request_ollama_json(
            service,
            "/api/generate",
            {
                "model": model,
                "stream": False,
                "keep_alive": service.keep_alive,
            },
        )
        service.loaded_models.add(model)

    results = []
    for prompt in prompt_lst:
        response = _request_ollama_json(
            service,
            "/api/chat",
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a helpful AI agent."},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "keep_alive": service.keep_alive,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
        )
        text = response.get("message", {}).get("content", "").strip()
        if not text:
            raise RuntimeError("Ollama returned an empty completion")
        if verbose:
            print(text)
        results.append(text)
    return results


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

    if isinstance(service, OllamaService):
        texts = query_ollama(
            service=service,
            model=model,
            prompt_lst=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            num_completions=num_completions,
            verbose=verbose,
        )
        return _as_request_result(texts)

    # DeepSeek, OpenAI, and Ollama all expose the OpenAI Chat Completions
    # interface. A non-HELM service that reaches this point uses that protocol.
    if not use_helm:
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
    ollama_model = _ollama_model_name(experiment_model)
    if ollama_model is not None:
        model_client = OllamaService(
            api_key=os.getenv("OLLAMA_API_KEY"),
        )
        return ollama_model, None, ollama_model, model_client

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
