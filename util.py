import json
import os
import time
from collections import namedtuple

import requests
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


_OUTLINES_MODEL_CACHE = {}
_GUIDANCE_MODEL_CACHE = {}


def _structured_value_to_text(value):
    if isinstance(value, str):
        return value
    if hasattr(value, "model_dump_json"):
        return value.model_dump_json()
    if hasattr(value, "json"):
        return value.json()
    return json.dumps(value, ensure_ascii=False)


def _generate_local_structured_json(
    model,
    tokenizer,
    prompts,
    schema,
    *,
    backend,
    fallback_backend,
    required,
    temperature,
    max_tokens,
):
    """Use Outlines/Guidance only for direct local-model constrained decoding."""
    backends = []
    for candidate in (backend, fallback_backend):
        normalized = str(candidate or "none").strip().lower()
        if normalized != "none" and normalized not in backends:
            backends.append(normalized)
    errors = []
    for candidate in backends:
        try:
            if candidate == "outlines":
                import outlines

                cache_key = (id(model), id(tokenizer))
                structured_model = _OUTLINES_MODEL_CACHE.get(cache_key)
                if structured_model is None:
                    structured_model = outlines.from_transformers(
                        model,
                        tokenizer,
                    )
                    _OUTLINES_MODEL_CACHE[cache_key] = structured_model
                call_kwargs = {"max_new_tokens": int(max_tokens)}
                if float(temperature) > 0:
                    call_kwargs["temperature"] = float(temperature)
                return [
                    _structured_value_to_text(
                        structured_model(prompt, schema, **call_kwargs)
                    )
                    for prompt in prompts
                ]
            if candidate == "guidance":
                from guidance import json as guidance_json
                from guidance import models as guidance_models

                cache_key = (id(model), id(tokenizer))
                guidance_model = _GUIDANCE_MODEL_CACHE.get(cache_key)
                if guidance_model is None:
                    guidance_model = guidance_models.Transformers(
                        model,
                        tokenizer=tokenizer,
                        echo=False,
                    )
                    _GUIDANCE_MODEL_CACHE[cache_key] = guidance_model
                results = []
                for prompt in prompts:
                    state = (
                        guidance_model
                        + prompt
                        + guidance_json(
                            name="structured_json",
                            schema=schema,
                            temperature=float(temperature),
                            max_tokens=int(max_tokens),
                        )
                    )
                    results.append(str(state["structured_json"]))
                return results
            errors.append(f"unsupported backend {candidate!r}")
        except Exception as exc:
            errors.append(
                f"{candidate}: {type(exc).__name__}: {exc}"
            )
    if required:
        raise RuntimeError(
            "Required local structured generation failed: "
            + "; ".join(errors or ["no backend configured"])
        )
    return None


def _transformers_dtype_kwargs(transformers_module, dtype):
    """Use the non-deprecated dtype keyword on Transformers 5+."""
    version_text = str(getattr(transformers_module, "__version__", "0"))
    try:
        major_version = int(version_text.split(".", 1)[0])
    except ValueError:
        major_version = 0
    return {"dtype": dtype} if major_version >= 5 else {"torch_dtype": dtype}


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
    stop_sequences=None,
    top_p=1.0,
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
                    "top_p": top_p,
                    "num_predict": max_tokens,
                    "stop": list(stop_sequences or []),
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
        modelpath,
        low_cpu_mem_usage=True,
        **_transformers_dtype_kwargs(transformers, torch.float16),
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
    import transformers

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **_transformers_dtype_kwargs(transformers, torch.bfloat16),
    ).eval()
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


def _truncate_at_stop(text, stop_sequences):
    positions = [
        position
        for stop_sequence in stop_sequences or []
        if stop_sequence
        for position in [text.find(stop_sequence)]
        if position >= 0
    ]
    return text[:min(positions)].rstrip() if positions else text


def _contains_complete_structured_json(text):
    required_fields = (
        '"reasoning_summary"',
        '"final_answer"',
        '"answer_type"',
        '"confidence"',
    )
    for start in (index for index, character in enumerate(text) if character == "{"):
        depth = 0
        quoted = False
        escaped = False
        for index in range(start, len(text)):
            character = text[index]
            if quoted:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    quoted = False
                continue
            if character == '"':
                quoted = True
            elif character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    fragment = text[start:index + 1]
                    if all(field in fragment for field in required_fields):
                        return True
                    break
    return False


def gen_from_prompt(
    model,
    tokenizer,
    prompt,
    echo_prompt=False,
    temperature=0.0,
    top_p=1.0,
    max_tokens=20,
    num_completions=1,
    output_scores=False,
    service=None,
    seed=101,
    process_func=None,
    terminate_by_linebreak=True,
    stop_sequences=None,
    verbose=False,
    use_helm=False,
    auth=None,
    structured_schema=None,
    structured_backend="none",
    structured_fallback_backend="none",
    structured_required=False,
    budget_role="other",
    request_timeout_seconds=None,
    max_num_retries=5,
    retry_delay_seconds=10,
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
        formatted_prompts = list(prompt)
        if getattr(tokenizer, "chat_template", None):
            formatted_prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": item}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for item in prompt
            ]
        if structured_schema is not None:
            structured_texts = _generate_local_structured_json(
                model,
                tokenizer,
                formatted_prompts,
                structured_schema,
                backend=structured_backend,
                fallback_backend=structured_fallback_backend,
                required=bool(structured_required),
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if structured_texts is not None:
                return _as_request_result(structured_texts)
        prompt_ids = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
        )
        attention_mask = prompt_ids["attention_mask"].to(model.device)
        input_ids = prompt_ids["input_ids"].to(model.device)
        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "do_sample": temperature > 0,
            "max_new_tokens": max_tokens,
            "num_return_sequences": num_completions,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": (
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id
            ),
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p
        encoded_stops = []
        for stop_sequence in stop_sequences or []:
            stop_ids = tokenizer(
                stop_sequence,
                add_special_tokens=False,
            ).input_ids
            if stop_ids:
                encoded_stops.append(stop_ids)
        if encoded_stops:
            from transformers import StoppingCriteria, StoppingCriteriaList

            prompt_length = input_ids.size(1)

            class StopOnSequences(StoppingCriteria):
                def __call__(self, generated, scores, **kwargs):
                    del scores, kwargs
                    suffixes = generated[:, prompt_length:]
                    return all(
                        (
                            any(
                                row.size(0) >= len(stop_ids)
                                and row[-len(stop_ids):].tolist() == stop_ids
                                for stop_ids in encoded_stops
                            )
                            or (
                                row.size(0) > 0
                                and "}" in tokenizer.decode(
                                    row[-2:],
                                    skip_special_tokens=False,
                                )
                                and _contains_complete_structured_json(
                                    tokenizer.decode(
                                        row,
                                        skip_special_tokens=True,
                                    )
                                )
                            )
                        )
                        for row in suffixes
                    )

            generation_kwargs["stopping_criteria"] = StoppingCriteriaList(
                [StopOnSequences()]
            )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            generated_ids = model.generate(**generation_kwargs)
        texts = tokenizer.batch_decode(
            generated_ids[:, input_ids.size(1) :], skip_special_tokens=True
        )
        if stop_sequences:
            texts = [
                _truncate_at_stop(text, stop_sequences)
                for text in texts
            ]
        if terminate_by_linebreak != "no":
            texts = [text.split("\n")[0] for text in texts]
        return _as_request_result(texts)

    if model.startswith("claude"):
        texts = query_claude(
            client=service,
            model=model,
            prompt_lst=prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            num_completions=num_completions,
            verbose=verbose,
            stop_sequences=stop_sequences,
        )
        return _as_request_result(texts)

    if isinstance(service, OllamaService):
        texts = query_ollama(
            service=service,
            model=model,
            prompt_lst=prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            num_completions=num_completions,
            verbose=verbose,
            stop_sequences=stop_sequences,
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
            top_p=top_p,
            max_tokens=max_tokens,
            num_completions=num_completions,
            verbose=verbose,
            stop_sequences=stop_sequences,
            request_timeout_seconds=request_timeout_seconds,
            max_num_retries=max_num_retries,
            retry_delay_seconds=retry_delay_seconds,
            budget_role=budget_role,
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
                    stop_sequences=list(stop_sequences or ["\n"]),
                )
                return service.make_request(auth, request)
            except Exception:
                if retry == 4:
                    raise
                time.sleep(10)
    raise NotImplementedError(f"Unsupported model: {model}")


def query_claude(
    client,
    model,
    prompt_lst,
    temperature,
    max_tokens,
    num_completions,
    verbose,
    stop_sequences=None,
    max_num_retries=5,
    top_p=1.0,
):
    results = []
    for prompt in prompt_lst:
        message = None
        for retry in range(max_num_retries):
            try:
                request_kwargs = {
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "messages": [{"role": "user", "content": prompt}],
                    "model": model,
                }
                if stop_sequences:
                    request_kwargs["stop_sequences"] = list(stop_sequences)
                message = client.messages.create(
                    **request_kwargs
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
    client,
    model,
    prompt_lst,
    temperature,
    max_tokens,
    num_completions,
    verbose,
    stop_sequences=None,
    max_num_retries=5,
    top_p=1.0,
    request_timeout_seconds=None,
    retry_delay_seconds=10,
    budget_role="other",
):
    results = []
    retry_count = max(1, int(max_num_retries))
    retry_delay = max(0.0, float(retry_delay_seconds))
    timeout_seconds = (
        None
        if request_timeout_seconds is None
        else float(request_timeout_seconds)
    )
    for prompt_index, prompt in enumerate(prompt_lst, start=1):
        completion = None
        for retry in range(retry_count):
            started = time.monotonic()
            timeout_label = (
                "sdk_default"
                if timeout_seconds is None
                else f"{timeout_seconds:g}s"
            )
            print(
                "[API] request_start "
                f"model={model} prompt={prompt_index}/{len(prompt_lst)} "
                f"attempt={retry + 1}/{retry_count} "
                f"timeout={timeout_label}",
                flush=True,
            )
            ledger = None
            if budget_role in {"generation", "validation"}:
                from autobencher.budget_ledger import active_ledger

                ledger = active_ledger()
                if ledger is not None and budget_role == "generation":
                    from autobencher.budget_ledger import estimate_tokens

                    ledger.assert_generation_available(
                        reserved_input_tokens=estimate_tokens(prompt),
                        reserved_output_tokens=int(max_tokens),
                    )
            try:
                request_kwargs = dict(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful AI agent."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    top_p=top_p,
                    n=num_completions,
                )
                # A declared output reserve is enforceable only if the same
                # cap is sent to every provider, including DeepSeek-compatible
                # endpoints.
                request_kwargs["max_tokens"] = max_tokens
                if stop_sequences:
                    request_kwargs["stop"] = list(stop_sequences)
                if timeout_seconds is not None:
                    request_kwargs["timeout"] = timeout_seconds
                completion = client.chat.completions.create(**request_kwargs)
                content = completion.choices[0].message.content
                if not content or not content.strip():
                    raise ValueError("API returned an empty completion")
                if ledger is not None:
                    from autobencher.budget_ledger import estimate_tokens

                    usage = getattr(completion, "usage", None)
                    input_tokens = (
                        getattr(usage, "prompt_tokens", None)
                        or getattr(usage, "input_tokens", None)
                    )
                    output_tokens = (
                        getattr(usage, "completion_tokens", None)
                        or getattr(usage, "output_tokens", None)
                    )
                    exact_tokens = (
                        input_tokens is not None
                        and output_tokens is not None
                    )
                    ledger.record_provider_call(
                        role=budget_role,
                        input_tokens=(
                            int(input_tokens)
                            if input_tokens is not None
                            else estimate_tokens(prompt)
                        ),
                        output_tokens=(
                            int(output_tokens)
                            if output_tokens is not None
                            else estimate_tokens(content)
                        ),
                        wall_time_seconds=time.monotonic() - started,
                        retry=retry > 0,
                        api_calls=1,
                        exact_tokens=exact_tokens,
                    )
                print(
                    "[API] request_done "
                    f"model={model} prompt={prompt_index}/{len(prompt_lst)} "
                    f"attempt={retry + 1}/{retry_count} "
                    f"elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )
                break
            except Exception as exc:
                elapsed = time.monotonic() - started
                if budget_role in {"generation", "validation"} and ledger is not None:
                    from autobencher.budget_ledger import estimate_tokens

                    ledger.record_provider_call(
                        role=budget_role,
                        input_tokens=estimate_tokens(prompt),
                        output_tokens=0,
                        wall_time_seconds=elapsed,
                        retry=retry > 0,
                        api_calls=1,
                        exact_tokens=False,
                        success=False,
                    )
                if retry == retry_count - 1:
                    raise RuntimeError(
                        f"API request failed after {retry_count} attempts "
                        f"(last attempt {elapsed:.1f}s): {exc}"
                    ) from exc
                print(
                    "[API] request_retry "
                    f"model={model} attempt={retry + 1}/{retry_count} "
                    f"elapsed={elapsed:.1f}s "
                    f"error={type(exc).__name__} "
                    f"sleep={retry_delay:g}s",
                    flush=True,
                )
                time.sleep(retry_delay)
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
