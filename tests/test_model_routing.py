import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests
import run_scripts
import util


class OllamaRoutingTests(unittest.TestCase):
    def test_completion_is_truncated_at_first_stop_sequence(self):
        text = '{"final_answer":"4"}Human: unrelated'
        self.assertEqual(
            util._truncate_at_stop(text, ["Human:", "User:"]),
            '{"final_answer":"4"}',
        )

    def test_complete_structured_json_is_detected_after_preamble(self):
        text = (
            "Reasoning first.\n"
            '{"reasoning_summary":["Add."],"final_answer":"4",'
            '"answer_type":"integer","confidence":1.0}'
            "Human: unrelated"
        )
        self.assertTrue(util._contains_complete_structured_json(text))

    def test_openai_compatible_request_receives_stop_sequences(self):
        client = Mock()
        completion = Mock()
        completion.choices = [
            Mock(message=Mock(content='{"final_answer":"4"}'))
        ]
        client.chat.completions.create.return_value = completion

        util.gen_from_prompt(
            model="local-openai-compatible",
            tokenizer=None,
            prompt=["Return JSON."],
            service=client,
            stop_sequences=["Human:", "<|im_end|>"],
        )

        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["stop"], ["Human:", "<|im_end|>"])

    def test_transformers_five_uses_dtype_keyword(self):
        module = Mock(__version__="5.14.1")
        marker = object()
        self.assertEqual(
            util._transformers_dtype_kwargs(module, marker),
            {"dtype": marker},
        )

    def test_transformers_four_uses_legacy_dtype_keyword(self):
        module = Mock(__version__="4.49.0")
        marker = object()
        self.assertEqual(
            util._transformers_dtype_kwargs(module, marker),
            {"torch_dtype": marker},
        )

    def test_ollama_tag_uses_native_local_endpoint(self):
        with patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://localhost:11434"}, clear=False):
            model, tokenizer, name, client = util.process_args_for_models(
                "qwen2.5:7b-instruct"
            )

        self.assertEqual(model, "qwen2.5:7b-instruct")
        self.assertIsNone(tokenizer)
        self.assertEqual(name, model)
        self.assertIsInstance(client, util.OllamaService)
        self.assertEqual(client.base_url, "http://localhost:11434")

    @patch.object(util, "query_ollama", return_value=["4"])
    def test_ollama_service_is_used_for_generation(self, query):
        service = util.OllamaService()
        result = util.gen_from_prompt(
            model="qwen2.5:7b-instruct",
            tokenizer=None,
            prompt=["2 + 2"],
            service=service,
        )

        self.assertEqual(result.completions[0].text, "4")
        self.assertIs(query.call_args.kwargs["service"], service)

    @patch.object(util.time, "sleep")
    def test_transient_ollama_failure_is_retried(self, sleep):
        service = util.OllamaService()
        service.max_retries = 2
        service.retry_delay = 0.1
        successful_response = Mock()
        successful_response.raise_for_status.return_value = None
        successful_response.json.return_value = {"message": {"content": "4"}}
        service.session.post = Mock(
            side_effect=[requests.ConnectionError("model runner unavailable"), successful_response]
        )

        result = util._request_ollama_json(
            service,
            "/api/chat",
            {"model": "qwen2.5:7b-instruct"},
        )

        self.assertEqual(result["message"]["content"], "4")
        self.assertEqual(service.session.post.call_count, 2)
        sleep.assert_called_once_with(0.1)

    @patch.object(util, "_request_ollama_json")
    def test_native_query_preloads_and_keeps_model_alive(self, request):
        request.side_effect = [
            {"done": True},
            {"message": {"content": "4"}},
            {"message": {"content": "6"}},
        ]
        service = util.OllamaService()

        first = util.query_ollama(
            service, "qwen2.5:7b-instruct", ["2+2"], 0.01, 50, 1, False
        )
        second = util.query_ollama(
            service, "qwen2.5:7b-instruct", ["3+3"], 0.01, 50, 1, False
        )

        self.assertEqual(first, ["4"])
        self.assertEqual(second, ["6"])
        self.assertEqual(request.call_count, 3)
        self.assertEqual(request.call_args_list[0].args[1], "/api/generate")
        self.assertEqual(request.call_args_list[1].args[1], "/api/chat")


class LauncherTests(unittest.TestCase):
    def test_separate_agent_and_test_taker_options_are_forwarded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            prefix = str(Path(temp_dir, "math_test", "result."))
            command = run_scripts.build_command(
                "math",
                "fallback",
                2,
                agent_modelname="deepseek-v4-pro",
                test_taker_modelname="qwen2.5:7b-instruct",
                outfile_prefix1=prefix,
                acc_target="0.1--0.3",
            )

            self.assertIn("deepseek-v4-pro", command)
            self.assertIn("qwen2.5:7b-instruct", command)
            self.assertEqual(command[command.index("--num_iters") + 1], "2")
            self.assertEqual(command[command.index("--outfile_prefix1") + 1], prefix)
            self.assertTrue(Path(prefix).parent.is_dir())

    def test_main_accepts_original_underscore_style_command(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            prefix = str(Path(temp_dir, "math_test", "result."))
            argv = [
                "run_scripts.py",
                "math",
                "--agent_modelname",
                "deepseek-v4-pro",
                "--test_taker_modelname",
                "qwen2.5:7b-instruct",
                "--exp_mode",
                "autobencher",
                "--use_helm",
                "no",
                "--num_iters",
                "2",
                "--outfile_prefix1",
                prefix,
                "--acc_target",
                "0.1--0.3",
            ]
            with patch.object(sys, "argv", argv), patch.object(
                run_scripts.subprocess, "run"
            ) as subprocess_run:
                run_scripts.main()

            command = subprocess_run.call_args.args[0]
            self.assertEqual(
                command[command.index("--agent_modelname") + 1],
                "deepseek-v4-pro",
            )
            self.assertEqual(
                command[command.index("--test_taker_modelname") + 1],
                "qwen2.5:7b-instruct",
            )
            subprocess_run.assert_called_once_with(command, check=True)


if __name__ == "__main__":
    unittest.main()
