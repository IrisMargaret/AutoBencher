import ast
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import math_autobencher
import tool_util
from autobencher.config import load_project_config, load_resolved_config
from autobencher.truth_solver import FailureType
from tool_util import (
    ERROR_TAGS,
    clean_redundant_files,
    dump_standard_json,
    extract_json_v2,
    manage_hard_pool,
)

ROOT = Path(__file__).resolve().parents[1]


class SemanticJudgmentResilienceTests(unittest.TestCase):
    @staticmethod
    def _success(reason):
        return {
            "semantically_equivalent": True,
            "confidence": 0.99,
            "reason": reason,
            "format_only_difference": False,
            "status": "success",
            "attempt": 1,
            "prompt_version": "test",
            "prompt_sha256": "abc",
            "deterministic_equivalent": True,
        }

    def test_provider_failure_is_checkpointed_and_only_failure_retries(self):
        config = load_resolved_config(
            ROOT / "configs" / "math_flywheel_smoke_test.yaml"
        )[0]
        gold = [
            {"question": f"Compute {index} + 1.", "answer": str(index + 1)}
            for index in range(3)
        ]
        predicted = [
            {
                "test_taker_response": str(index + 1),
                "answer_type": "integer",
            }
            for index in range(3)
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "judge.compare_answers.json"
            with patch(
                "math_autobencher.judge_answer_semantics",
                side_effect=[
                    self._success("first"),
                    RuntimeError("API returned an empty completion"),
                    self._success("third"),
                ],
            ):
                first = math_autobencher._evaluate_semantic_judgments(
                    gold,
                    predicted,
                    tool_info=("model", None, object()),
                    research_config=config,
                    judge_cache_path=cache_path,
                )

            self.assertEqual(len(first), 3)
            self.assertEqual(first[0]["semantic_judge"]["status"], "success")
            self.assertEqual(first[1]["semantic_judge"]["status"], "failed")
            self.assertEqual(first[2]["semantic_judge"]["status"], "success")
            self.assertEqual(len(json.loads(cache_path.read_text("utf-8"))), 3)

            with patch(
                "math_autobencher.judge_answer_semantics",
                return_value=self._success("recovered"),
            ) as judge:
                resumed = math_autobencher._evaluate_semantic_judgments(
                    gold,
                    predicted,
                    tool_info=("model", None, object()),
                    research_config=config,
                    judge_cache_path=cache_path,
                )

            judge.assert_called_once()
            self.assertTrue(
                all(
                    item["semantic_judge"]["status"] == "success"
                    for item in resumed
                )
            )
            self.assertEqual(resumed[0]["reasons"], "first")
            self.assertEqual(resumed[1]["reasons"], "recovered")
            self.assertEqual(resumed[2]["reasons"], "third")

    def test_thread_safe_client_uses_parallel_ordered_checkpointing(self):
        config = load_resolved_config(
            ROOT / "configs" / "math_flywheel_smoke_test.yaml"
        )[0]
        gold = [
            {"question": f"Compute {index} + 1.", "answer": str(index + 1)}
            for index in range(6)
        ]
        predicted = [
            {
                "test_taker_response": str(index + 1),
                "answer_type": "integer",
            }
            for index in range(6)
        ]
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=object())
        )
        lock = threading.Lock()
        active = 0
        maximum_active = 0

        def judge(**kwargs):
            nonlocal active, maximum_active
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.02)
            with lock:
                active -= 1
            return self._success(kwargs["question"])

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "parallel.compare_answers.json"
            with patch(
                "math_autobencher.judge_answer_semantics",
                side_effect=judge,
            ) as semantic_judge:
                first = math_autobencher._evaluate_semantic_judgments(
                    gold,
                    predicted,
                    tool_info=("model", None, client),
                    research_config=config,
                    judge_cache_path=cache_path,
                )

            self.assertGreaterEqual(maximum_active, 2)
            self.assertEqual(semantic_judge.call_count, 6)
            self.assertEqual(
                [item["reasons"] for item in first],
                [item["question"] for item in gold],
            )
            self.assertEqual(
                json.loads(cache_path.read_text(encoding="utf-8")),
                first,
            )

            with patch(
                "math_autobencher.judge_answer_semantics"
            ) as resumed_judge:
                resumed = math_autobencher._evaluate_semantic_judgments(
                    gold,
                    predicted,
                    tool_info=("model", None, client),
                    research_config=config,
                    judge_cache_path=cache_path,
                )
            resumed_judge.assert_not_called()
            self.assertEqual(resumed, first)


def test_parse_failure_does_not_read_unassigned_judge_conflict(tmp_path):
    config = load_resolved_config(
        ROOT / "configs" / "math_flywheel_smoke_test.yaml"
    )[0]
    inference = {
        "id": 1,
        "question_id": "parse-failure-1",
        "category": "Arithmetic",
        "sub_category": "Integer Operations",
        "difficulty": 2,
        "question": "Compute 2 + 2.",
        "gold_answer": "4",
        "canonical_answer": "4",
        "display_answer": "4",
        "answer_type": "integer",
        "test_taker_response": "",
        "parse_status": "parse_failed",
        "parsed_response": {},
        "parser_version": "structured_v2",
    }
    judgment = {
        "question": inference["question"],
        "gold_answer": "4",
        "test_taker_answer": "",
        "is_correct": False,
        "confidence": 0.0,
        "reasons": "invalid structured response",
        "semantic_judge": {"status": "failed"},
    }
    prefix = str(tmp_path / "parse_failure")
    with patch.object(
        math_autobencher,
        "generate_math_inference",
        return_value=[inference],
    ), patch.object(
        math_autobencher,
        "_evaluate_semantic_judgments",
        return_value=[judgment],
    ):
        records = math_autobencher.test_and_eval(
            [inference],
            prefix,
            test_taker_info=None,
            agent_info=None,
            tool_info=None,
            research_config=config,
        )

    assert records[0]["is_correct"] is False
    assert records[0]["evaluation_status"] == "parse_failed"
    assert isinstance(records[0]["equivalence_needs_review"], bool)


def test_null_semantic_confidence_fails_closed_without_aborting(tmp_path):
    config = load_resolved_config(
        ROOT / "configs" / "math_flywheel_smoke_test.yaml"
    )[0]
    inference = {
        "id": 1,
        "question_id": "null-confidence-1",
        "category": "Arithmetic",
        "sub_category": "Integer Operations",
        "difficulty": 2,
        "question": "Compute 2 + 2.",
        "gold_answer": "4",
        "canonical_answer": "4",
        "display_answer": "4",
        "answer_type": "integer",
        "test_taker_response": "4",
        "parse_status": "success",
        "parsed_response": {"final_answer": "4"},
        "parser_version": "structured_v2",
    }
    judgment = {
        "question": inference["question"],
        "gold_answer": "4",
        "test_taker_answer": "4",
        "is_correct": True,
        "confidence": None,
        "reasons": "provider omitted confidence",
        "semantic_judge": {"status": "success"},
    }
    prefix = str(tmp_path / "null_confidence")
    with patch.object(
        math_autobencher,
        "generate_math_inference",
        return_value=[inference],
    ), patch.object(
        math_autobencher,
        "_evaluate_semantic_judgments",
        return_value=[judgment],
    ):
        records = math_autobencher.test_and_eval(
            [inference],
            prefix,
            test_taker_info=None,
            agent_info=None,
            tool_info=None,
            research_config=config,
        )

    assert records[0]["is_correct"] is True
    assert records[0]["evaluator_confidence"] == 0.0


def test_null_confidence_resume_reuses_inference_and_judge_caches(tmp_path):
    config = load_resolved_config(
        ROOT / "configs" / "math_flywheel_smoke_test.yaml"
    )[0]
    inference = {
        "id": 1,
        "question_id": "resumed-null-confidence-1",
        "category": "Arithmetic",
        "sub_category": "Integer Operations",
        "difficulty": 2,
        "question": "Compute 2 + 2.",
        "gold_answer": "4",
        "canonical_answer": "4",
        "display_answer": "4",
        "answer_type": "integer",
        "test_taker_response": "4",
        "parse_status": "success",
        "parsed_response": {"final_answer": "4", "confidence": None},
        "parser_version": "structured_v2",
    }
    judgment = {
        "question": inference["question"],
        "gold_answer": "4",
        "test_taker_answer": "4",
        "is_correct": True,
        "confidence": None,
        "reasons": "provider omitted confidence",
        "semantic_judge": {"status": "success"},
    }
    prefix = tmp_path / "resumed_null_confidence"
    inference_path = Path(f"{prefix}.test_taker_inference.json")
    judge_path = tmp_path / "temp_log" / "judge.compare_answers.json"
    judge_path.parent.mkdir(parents=True)
    dump_standard_json([inference], inference_path)
    dump_standard_json([judgment], judge_path)

    with patch("tool_util.gen_from_prompt") as inference_api, patch.object(
        math_autobencher,
        "judge_answer_semantics",
    ) as judge_api:
        records = math_autobencher.test_and_eval(
            [inference],
            str(prefix),
            test_taker_info=("cached-model", None, None),
            agent_info=None,
            tool_info=("cached-judge", None, None),
            research_config=config,
            temp_log_dir=str(judge_path.parent),
        )

    inference_api.assert_not_called()
    judge_api.assert_not_called()
    assert records[0]["is_correct"] is True
    assert records[0]["evaluator_confidence"] == 0.0
    assert Path(f"{prefix}.compare_answers.json").is_file()


def test_attribution_exception_abstains_without_aborting_fixed_evaluation(
    tmp_path,
):
    config = load_resolved_config(
        ROOT / "configs" / "math_flywheel_smoke_test.yaml"
    )[0]
    inference = {
        "id": 1,
        "question_id": "attribution-failure-1",
        "category": "Linear Algebra",
        "sub_category": "Matrix Operations",
        "difficulty": 3,
        "question": "Compute det(Matrix([[1, 2], [3, 4]])).",
        "gold_answer": "-2",
        "canonical_answer": "-2",
        "display_answer": "-2",
        "answer_type": "integer",
        "test_taker_response": "0",
        "parse_status": "success",
        "parsed_response": {"final_answer": "0", "confidence": 0.5},
        "parser_version": "structured_v2",
    }
    judgment = {
        "question": inference["question"],
        "gold_answer": "-2",
        "test_taker_answer": "0",
        "is_correct": False,
        "confidence": 1.0,
        "reasons": "wrong answer",
        "semantic_judge": {"status": "success"},
    }
    prefix = str(tmp_path / "attribution_failure")
    with patch.object(
        math_autobencher,
        "generate_math_inference",
        return_value=[inference],
    ), patch.object(
        math_autobencher,
        "_evaluate_semantic_judgments",
        return_value=[judgment],
    ), patch.object(
        math_autobencher,
        "attribute_error",
        side_effect=TypeError("malformed normalized component"),
    ):
        records = math_autobencher.test_and_eval(
            [inference],
            prefix,
            test_taker_info=None,
            agent_info=None,
            tool_info=None,
            research_config=config,
        )

    assert records[0]["is_correct"] is False
    assert records[0]["primary_error_tag"] == "unknown_error"
    assert records[0]["verification_tier"] == "abstained"
    assert Path(f"{prefix}.compare_answers.json").is_file()


def test_checkpoint_manifest_has_one_authoritative_writer():
    tree = ast.parse(
        (ROOT / "math_autobencher.py").read_text(encoding="utf-8")
    )
    writes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "atomic_json"
        and len(node.args) >= 2
        and "checkpoint_manifest.json" in ast.unparse(node.args[1])
    ]
    assert len(writes) == 1


class ExtractJsonTests(unittest.TestCase):
    def test_extracts_json_from_supported_model_formats(self):
        expected = [[{"id": "1", "question": "1 + 1", "answer": "2"}]]
        responses = [
            '```json\n[{"id": "1", "question": "1 + 1", "answer": "2"}]\n```',
            '```\n[{"id": "1", "question": "1 + 1", "answer": "2"}]\n```',
            '[{"id": "1", "question": "1 + 1", "answer": "2"}]',
            'Here is the result:\n[{"id": "1", "question": "1 + 1", "answer": "2"}]\nDone.',
        ]
        for response in responses:
            with self.subTest(response=response):
                self.assertEqual(extract_json_v2(response, None), expected)

    def test_writes_cache_atomically(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir, "result.json")
            parsed = extract_json_v2('```json\n[{"id": 1}]\n```', output)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), parsed)
            self.assertFalse(Path(f"{output}.tmp").exists())


class MathPlanRetryTests(unittest.TestCase):
    def test_invalid_response_is_retried_and_valid_plan_is_cached(self):
        completion = type("Completion", (), {})
        first = completion()
        first.text = "I cannot provide JSON."
        second = completion()
        plan_items = [
            {
                "id": str(index),
                "category": "Arithmetic",
                "subcategory_description": f"subcategory {index}",
                "difficulty": "2",
            }
            for index in range(1, 6)
        ]
        second.text = f"```json\n{json.dumps(plan_items)}\n```"
        request_result = type("RequestResult", (), {})
        first_result = request_result()
        first_result.completions = [first]
        second_result = request_result()
        second_result.completions = [second]

        with tempfile.TemporaryDirectory() as temp_dir:
            prefix = str(Path(temp_dir, "iteration."))
            with patch.object(
                math_autobencher,
                "gen_from_prompt",
                side_effect=[first_result, second_result],
            ) as generate:
                result = math_autobencher._generate_cat_with_aim(
                    "0.1--0.3",
                    "model",
                    None,
                    object(),
                    [],
                    1,
                    prefix,
                )

            self.assertEqual(generate.call_count, 2)
            self.assertEqual(result[0][0]["category"], "Arithmetic")
            cache = Path(f"{prefix}.question_plan_with_aim.json")
            self.assertEqual(
                json.loads(cache.read_text(encoding="utf-8")), result[0]
            )
            self.assertTrue(
                Path(
                    temp_dir,
                    "temp_log",
                    f"{Path(prefix).name}.question_plan_with_aim.attempt1.txt",
                ).exists()
            )

    def test_oversized_plan_is_limited_to_requested_five_items(self):
        items = [
            {
                "id": str(index),
                "category": "Arithmetic",
                "subcategory_description": f"subcategory {index}",
            }
            for index in range(1, 8)
        ]
        self.assertEqual(
            len(math_autobencher._normalize_math_plan([items])[0]),
            5,
        )


class GoldAnswerValidationTests(unittest.TestCase):
    @staticmethod
    def _result(payload):
        completion = type("Completion", (), {})()
        completion.text = json.dumps(payload)
        result = type("RequestResult", (), {})()
        result.completions = [completion]
        return result

    def test_only_independently_verified_gold_answers_are_accepted(self):
        config, _ = load_project_config(
            ROOT / "configs" / "math_flywheel_smoke_test.yaml"
        )
        questions = [
            {
                "question_id": "q1",
                "question": "What is 2 + 2?",
                "answer_type": "integer",
                "canonical_answer": "4",
                "unit": None,
                "tolerance": None,
            },
            {
                "question_id": "q2",
                "question": "What is 3 + 3?",
                "answer_type": "integer",
                "canonical_answer": "7",
                "unit": None,
                "tolerance": None,
            },
        ]
        validation = [
            {
                "validation_id": 0,
                "recomputed_answer": "4",
                "answer_type": "integer",
                "verification_passed": True,
                "substitution_passed": True,
                "verification_method": "independent addition",
                "failure_reason": None,
            },
            {
                "validation_id": 1,
                "recomputed_answer": "6",
                "answer_type": "integer",
                "verification_passed": False,
                "substitution_passed": False,
                "verification_method": "independent addition",
                "failure_reason": "The proposed answer is incorrect.",
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            prefix = str(Path(temp_dir, "iteration.subcat0"))
            with patch.object(
                math_autobencher,
                "gen_from_prompt",
                return_value=self._result(validation),
            ):
                accepted = (
                    math_autobencher._validate_generated_gold_answers(
                        questions,
                        "model",
                        None,
                        object(),
                        config,
                        prefix,
                    )
                )

            self.assertEqual(
                [item["question_id"] for item in accepted],
                ["q1"],
            )
            self.assertEqual(
                accepted[0]["gold_answer_validation"]["status"],
                "passed",
            )
            canonical = tool_util.canonicalize_math_record(
                accepted[0],
                0,
            )
            self.assertEqual(
                canonical["gold_answer_validation"]["status"],
                "passed",
            )
            audit_path = Path(
                f"{prefix}.gold_answer_validation.json"
            )
            audits = json.loads(audit_path.read_text(encoding="utf-8"))
            self.assertEqual(len(audits), 2)
            self.assertEqual(
                audits[1]["validation"]["status"],
                "failed",
            )
            clean_redundant_files(temp_dir)
            self.assertTrue(audit_path.is_file())

    def test_string_boolean_is_rejected_and_retried(self):
        config, _ = load_project_config(
            ROOT / "configs" / "math_flywheel_smoke_test.yaml"
        )
        invalid = [
            {
                "validation_id": 0,
                "recomputed_answer": "4",
                "answer_type": "integer",
                "verification_passed": "false",
                "substitution_passed": True,
                "verification_method": "independent addition",
                "failure_reason": None,
            }
        ]
        valid = [
            {
                "validation_id": 0,
                "recomputed_answer": "4",
                "answer_type": "integer",
                "verification_passed": True,
                "substitution_passed": True,
                "verification_method": "independent addition",
                "failure_reason": None,
            }
        ]
        question = {
            "question_id": "q1",
            "question": "What is 2 + 2?",
            "answer_type": "integer",
            "canonical_answer": "4",
            "unit": None,
            "tolerance": None,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            prefix = str(Path(temp_dir, "iteration.subcat0"))
            with patch.object(
                math_autobencher,
                "gen_from_prompt",
                side_effect=[
                    self._result(invalid),
                    self._result(valid),
                ],
            ) as generate:
                accepted = (
                    math_autobencher._validate_generated_gold_answers(
                        [question],
                        "model",
                        None,
                        object(),
                        config,
                        prefix,
                    )
                )
            self.assertEqual(generate.call_count, 2)
            self.assertEqual(len(accepted), 1)

    def test_wrong_tuple_system_gold_cannot_be_model_approved(self):
        config, _ = load_project_config(
            ROOT / "configs" / "math_flywheel_smoke_test.yaml"
        )
        question_text = (
            "Solve the following system of equations for (x, y, z): "
            "x + 2y - z = 5, 2x - y + 3z = 4, "
            "-x + 3y + 2z = 7."
        )
        question = {
            "question_id": "q_1",
            "question": question_text,
            "answer_type": "ordered_tuple",
            "canonical_answer": "(2, 1, -1)",
            "unit": None,
            "tolerance": None,
        }
        # Simulate the exact faulty evaluator response: every model-reported
        # field claims success and simply repeats the proposed answer.
        claimed_pass = [
            {
                "validation_id": 0,
                "recomputed_answer": "(2, 1, -1)",
                "answer_type": "ordered_tuple",
                "verification_passed": True,
                "substitution_passed": True,
                "verification_method": (
                    "independent recomputation and substitution"
                ),
                "failure_reason": None,
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            prefix = str(Path(temp_dir, "iteration.subcat0"))
            with patch.object(
                math_autobencher,
                "gen_from_prompt",
                return_value=self._result(claimed_pass),
            ):
                accepted = (
                    math_autobencher._validate_generated_gold_answers(
                        [question],
                        "model",
                        None,
                        object(),
                        config,
                        prefix,
                    )
                )

            self.assertEqual(accepted, [])
            audits = json.loads(
                Path(
                    f"{prefix}.gold_answer_validation.json"
                ).read_text(encoding="utf-8")
            )
            validation = audits[0]["validation"]
            self.assertFalse(validation["verification_passed"])
            self.assertFalse(validation["substitution_passed"])
            self.assertEqual(
                validation["recomputed_answer"],
                "(8/5, 11/5, 1)",
            )
            self.assertEqual(
                [
                    item["difference"]
                    for item in validation["substitution_details"]
                ],
                ["0", "-4", "-8"],
            )
            self.assertEqual(
                validation["source_question_sha256"],
                validation["solve_question_sha256"],
            )
            self.assertEqual(
                validation["source_question_sha256"],
                validation["substitution_question_sha256"],
            )


class GenerationQuotaRepairTests(unittest.TestCase):
    @staticmethod
    def _verified_question():
        return {
            "id": "q2",
            "question_id": "q2",
            "category": "Arithmetic",
            "subcategory": "Integer Operations",
            "sub_category": "Integer Operations",
            "difficulty": 2,
            "question": "What is 1 + 1?",
            "answer_type": "integer",
            "canonical_answer": "2",
            "display_answer": "2",
            "answer": "2",
            "gold_answer": "2",
            "unit": None,
            "tolerance": None,
            "order_sensitive": False,
            "generation_source": "coverage_deficit",
            "reference_hard_sample_ids": [],
            "target_error_type": None,
            "generation_strategy": "quota_repair",
            "gold_answer_validation": {
                "status": "passed",
                "verification_passed": True,
                "substitution_passed": True,
            },
        }

    def test_one_subcategory_shortfall_does_not_fail_iteration(self):
        config, _ = load_project_config(
            ROOT / "configs" / "math_flywheel_smoke_test.yaml"
        )
        plan = {
            "question_budget": 2,
            "cycle": 1,
            "global_iteration": 1,
            "allocations": [
                {
                    "category": "Algebra",
                    "sub_category": "Polynomials and Inequalities",
                    "question_count": 1,
                    "generation_source": "coverage_deficit",
                    "generation_strategy": "quota_repair",
                    "difficulty": 4,
                },
                {
                    "category": "Arithmetic",
                    "sub_category": "Integer Operations",
                    "question_count": 1,
                    "generation_source": "coverage_deficit",
                    "generation_strategy": "quota_repair",
                    "difficulty": 2,
                },
            ],
        }

        def generate(description, *args, **kwargs):
            del args, kwargs
            if description["sub_category"] == (
                "Polynomials and Inequalities"
            ):
                return [[]]
            return [[self._verified_question()]]

        with tempfile.TemporaryDirectory() as temp_dir:
            prefix = str(Path(temp_dir, "iteration"))
            with patch.object(
                math_autobencher,
                "_generate_question_text_with_truth",
                side_effect=generate,
            ) as generator:
                questions, _ = math_autobencher._ask_question_v3(
                    ("model", None, object()),
                    [],
                    1,
                    prefix,
                    generation_plan=plan,
                    research_config=config,
                )
            self.assertEqual(len(questions), 1)
            self.assertEqual(generator.call_count, 5)
            self.assertTrue(
                plan["generation_result"]["partial_iteration"]
            )
            self.assertEqual(
                plan["generation_result"]["question_shortfall"],
                1,
            )

    def test_all_single_sample_repairs_exhaust_without_runtime_error(self):
        config, _ = load_project_config(
            ROOT / "configs" / "math_flywheel_smoke_test.yaml"
        )
        plan = {
            "question_budget": 1,
            "cycle": 1,
            "global_iteration": 1,
            "allocations": [
                {
                    "category": "Algebra",
                    "sub_category": "Polynomials and Inequalities",
                    "question_count": 1,
                    "generation_source": "coverage_deficit",
                    "generation_strategy": "quota_repair",
                    "difficulty": 4,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            prefix = str(Path(temp_dir, "iteration"))
            with patch.object(
                math_autobencher,
                "_generate_question_text_with_truth",
                return_value=[[]],
            ):
                questions, _ = math_autobencher._ask_question_v3(
                    ("model", None, object()),
                    [],
                    1,
                    prefix,
                    generation_plan=plan,
                    research_config=config,
                )
        self.assertEqual(questions, [])
        self.assertTrue(plan["generation_result"]["below_minimum_verified"])
        self.assertEqual(
            plan["generation_result"]["question_shortfall"],
            1,
        )

    def test_completed_subcategory_checkpoint_is_reused_after_crash(self):
        config, _ = load_project_config(
            ROOT / "configs" / "math_flywheel_smoke_test.yaml"
        )
        plan = {
            "question_budget": 2,
            "cycle": 1,
            "global_iteration": 1,
            "allocations": [
                {
                    "category": "Arithmetic",
                    "sub_category": "Integer Operations",
                    "question_count": 1,
                    "generation_source": "coverage_deficit",
                    "generation_strategy": "quota_repair",
                    "difficulty": 2,
                },
                {
                    "category": "Arithmetic",
                    "sub_category": "Fraction and Decimal Operations",
                    "question_count": 1,
                    "generation_source": "coverage_deficit",
                    "generation_strategy": "quota_repair",
                    "difficulty": 2,
                },
            ],
        }
        first = self._verified_question()
        second = {
            **self._verified_question(),
            "id": "q3",
            "question_id": "q3",
            "question": "What is 1/2 + 1/4?",
            "subcategory": "Fraction and Decimal Operations",
            "sub_category": "Fraction and Decimal Operations",
            "answer_type": "rational",
            "canonical_answer": "3/4",
            "display_answer": "3/4",
            "answer": "3/4",
            "gold_answer": "3/4",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            prefix = str(Path(temp_dir, "iteration"))
            with patch.object(
                math_autobencher,
                "_generate_question_text_with_truth",
                side_effect=[[[first]], KeyboardInterrupt("simulated interruption")],
            ):
                with self.assertRaisesRegex(
                    KeyboardInterrupt,
                    "simulated interruption",
                ):
                    math_autobencher._ask_question_v3(
                        ("model", None, object()),
                        [],
                        1,
                        prefix,
                        generation_plan=plan,
                        research_config=config,
                    )

            checkpoint = Path(
                f"{prefix}.generation_subcategory_checkpoint.json"
            )
            self.assertTrue(checkpoint.is_file())
            with patch.object(
                math_autobencher,
                "_generate_question_text_with_truth",
                return_value=[[second]],
            ) as generator:
                questions, _ = math_autobencher._ask_question_v3(
                    ("model", None, object()),
                    [],
                    1,
                    prefix,
                    generation_plan=plan,
                    research_config=config,
                )

            self.assertEqual(generator.call_count, 1)
            self.assertEqual(len(questions), 2)
            self.assertEqual(
                [item["question"] for item in questions],
                [first["question"], second["question"]],
            )

    def test_generator_runtime_failure_becomes_subcategory_shortfall(self):
        config, _ = load_project_config(
            ROOT / "configs" / "math_flywheel_smoke_test.yaml"
        )
        plan = {
            "question_budget": 1,
            "cycle": 1,
            "global_iteration": 1,
            "allocations": [
                {
                    "category": "Arithmetic",
                    "sub_category": "Integer Operations",
                    "question_count": 1,
                    "generation_source": "coverage_deficit",
                    "generation_strategy": "quota_repair",
                    "difficulty": 2,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(
                math_autobencher,
                "_generate_question_text_with_truth",
                side_effect=RuntimeError("invalid generated batch"),
            ):
                questions, _ = math_autobencher._ask_question_v3(
                    ("model", None, object()),
                    [],
                    1,
                    str(Path(temp_dir, "iteration")),
                    generation_plan=plan,
                    research_config=config,
                )

        self.assertEqual(questions, [])
        result = plan["generation_result"]
        self.assertTrue(result["partial_iteration"])
        self.assertEqual(result["question_shortfall"], 1)
        self.assertEqual(
            result["subcategory_statistics"][0]["failure_counts"][
                FailureType.GENERATOR_FORMAT_ERROR.value
            ],
            4,
        )

    def test_truth_failure_does_not_block_replacement_candidate(self):
        config, _ = load_project_config(
            ROOT / "configs" / "math_flywheel_smoke_test.yaml"
        )
        plan = {
            "question_budget": 2,
            "cycle": 1,
            "global_iteration": 1,
            "allocations": [
                {
                    "category": "Arithmetic",
                    "sub_category": "Integer Operations",
                    "question_count": 2,
                    "generation_source": "coverage_deficit",
                    "generation_strategy": "quota_repair",
                    "difficulty": 2,
                }
            ],
        }
        first = self._verified_question()
        second = {
            **self._verified_question(),
            "question": "What is 2 + 2?",
            "canonical_answer": "4",
            "display_answer": "4",
            "answer": "4",
            "gold_answer": "4",
        }
        calls = []

        def generate(*args, **kwargs):
            calls.append(kwargs["question_count"])
            prefix = args[4]
            if len(calls) == 1:
                dump_standard_json(
                    {
                        "generator_output_questions": 2,
                        "failure_counts": {
                            FailureType.TRUTH_PARSE_FAIL.value: 1
                        },
                    },
                    f"{prefix}.generation_batch_summary.json",
                )
                return [[first]]
            dump_standard_json(
                {
                    "generator_output_questions": 1,
                    "failure_counts": {},
                },
                f"{prefix}.generation_batch_summary.json",
            )
            return [[second]]

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(
                math_autobencher,
                "_generate_question_text_with_truth",
                side_effect=generate,
            ):
                questions, _ = math_autobencher._ask_question_v3(
                    ("model", None, object()),
                    [],
                    1,
                    str(Path(temp_dir, "iteration")),
                    generation_plan=plan,
                    research_config=config,
                )
        self.assertEqual(calls, [2, 1])
        self.assertEqual(len(questions), 2)
        self.assertFalse(plan["generation_result"]["partial_iteration"])

    def test_repeated_subcategory_failure_triggers_temporary_cooldown(self):
        config, _ = load_project_config(
            ROOT / "configs" / "math_flywheel_smoke_test.yaml"
        )

        def plan(iteration):
            return {
                "question_budget": 1,
                "cycle": 1,
                "global_iteration": iteration,
                "allocations": [
                    {
                        "category": "Algebra",
                        "sub_category": "Polynomials and Inequalities",
                        "question_count": 1,
                        "generation_source": "coverage_deficit",
                        "generation_strategy": "quota_repair",
                        "difficulty": 4,
                    }
                ],
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            hard_pool = str(Path(temp_dir, "hard_pool.json"))
            with patch.object(
                math_autobencher,
                "_generate_question_text_with_truth",
                return_value=[[]],
            ) as generate:
                for iteration in range(1, 4):
                    math_autobencher._ask_question_v3(
                        ("model", None, object()),
                        [],
                        iteration,
                        str(Path(temp_dir, f"iter_{iteration}")),
                        hard_pool_file=hard_pool,
                        generation_plan=plan(iteration),
                        research_config=config,
                    )
                before_cooldown_calls = generate.call_count
                fourth_plan = plan(4)
                questions, _ = math_autobencher._ask_question_v3(
                    ("model", None, object()),
                    [],
                    4,
                    str(Path(temp_dir, "iter_4")),
                    hard_pool_file=hard_pool,
                    generation_plan=fourth_plan,
                    research_config=config,
                )
            self.assertEqual(questions, [])
            self.assertEqual(generate.call_count, before_cooldown_calls)
            self.assertEqual(
                fourth_plan["generation_result"][
                    "subcategory_shortfalls"
                ][0]["reason"],
                "subcategory_cooldown",
            )

    def test_cycle_generation_statistics_are_grouped_by_subcategory(self):
        summary = math_autobencher._aggregate_cycle_generation_statistics(
            [
                {
                    "subcategory_statistics": [
                        {
                            "category": "Algebra",
                            "sub_category": "Linear Equations",
                            "generated_total": 3,
                            "valid_samples": 2,
                            "failure_counts": {"truth_parse_fail": 1},
                            "coverage_gap": 1,
                        }
                    ]
                },
                {
                    "subcategory_statistics": [
                        {
                            "category": "Algebra",
                            "sub_category": "Linear Equations",
                            "generated_total": 2,
                            "valid_samples": 1,
                            "failure_counts": {"repair_exhausted": 1},
                            "coverage_gap": 1,
                        }
                    ]
                },
            ]
        )
        self.assertEqual(summary["generated_total"], 5)
        self.assertEqual(summary["valid_samples"], 3)
        self.assertEqual(summary["coverage_gap"], 2)
        self.assertEqual(
            summary["failure_counts"],
            {"repair_exhausted": 1, "truth_parse_fail": 1},
        )


class QuestionOnlyTruthPipelineTests(unittest.TestCase):
    @staticmethod
    def _result(payload):
        completion = type("Completion", (), {})()
        completion.text = json.dumps(payload)
        result = type("RequestResult", (), {})()
        result.completions = [completion]
        return result

    @staticmethod
    def _description(sub_category="Linear Equations"):
        return {
            "category": "Algebra",
            "sub_category": sub_category,
            "difficulty": 4,
            "generation_source": "coverage_deficit",
            "generation_strategy": "quota_repair",
        }

    def test_llm_question_only_output_receives_truth_solver_gold(self):
        config, _ = load_project_config(
            ROOT / "configs" / "math_flywheel_smoke_test.yaml",
            temporary_overrides=["evaluator_pipeline.enabled=false"],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            prefix = str(Path(temp_dir, "subcat0"))
            with (
                patch.object(
                    math_autobencher,
                    "gen_from_prompt",
                    return_value=self._result(
                        [{"question": "Solve for x: 3x + 2 = 11."}]
                    ),
                ) as generate,
                patch.object(
                    math_autobencher,
                    "solve_with_privileged_python",
                ) as llm_solver,
            ):
                result = (
                    math_autobencher._generate_question_text_with_truth(
                        self._description(),
                        "model",
                        None,
                        object(),
                        prefix,
                        question_count=1,
                        research_config=config,
                    )
                )
            question = result[0][0]
            self.assertEqual(question["canonical_answer"], "3")
            self.assertEqual(question["gold_answer"], "3")
            self.assertEqual(len(question["gold_reasoning_summary"]), 2)
            self.assertEqual(question["target_difficulty"], 4)
            self.assertEqual(
                question["observed_difficulty"],
                question["difficulty_profile"]["score"],
            )
            self.assertEqual(
                question["difficulty"],
                question["difficulty_profile"]["effective_score"],
            )
            self.assertEqual(
                question["difficulty_profile"]["rubric_version"],
                "observable_math_v1",
            )
            self.assertEqual(
                question["truth_validation_details"]["solver_backend"],
                "sympy",
            )
            llm_solver.assert_not_called()
            self.assertIsNone(question["failure_type"])
            self.assertEqual(
                question["truth_validation_details"]["solver_branch"],
                "single_equation",
            )
            self.assertEqual(
                generate.call_args.kwargs["temperature"],
                0.0,
            )
            self.assertEqual(generate.call_args.kwargs["top_p"], 0.1)
            self.assertIn(
                "Objective difficulty profile",
                generate.call_args.kwargs["prompt"][0],
            )
            summary = json.loads(
                Path(
                    f"{prefix}.generation_batch_summary.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(summary["gold_solver_backend"], "sympy")
            self.assertEqual(summary["llm_gold_solver_calls"], 0)

    def test_objective_difficulty_can_reject_out_of_band_question(self):
        config, _ = load_project_config(
            ROOT / "configs" / "math_flywheel_smoke_test.yaml",
            temporary_overrides=[
                "evaluator_pipeline.enabled=false",
                "generation.minimum_difficulty=2",
                "generation.maximum_difficulty=2",
                "evaluator_pipeline.minimum_difficulty=2",
                "evaluator_pipeline.maximum_difficulty=2",
                "adaptive_sampling.initial_difficulty=2",
                "difficulty.reject_outside_generation_bounds=true",
            ],
        )
        description = {
            "category": "Algebra",
            "sub_category": "Systems of Equations",
            "difficulty": 2,
            "generation_source": "coverage_deficit",
            "generation_strategy": "quota_repair",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            prefix = str(Path(temp_dir, "difficulty"))
            with patch.object(
                math_autobencher,
                "gen_from_prompt",
                return_value=self._result(
                    [
                        {
                            "question": (
                                "Solve the system for (x, y): "
                                "x + y = 7, 2*x - y = 2."
                            )
                        }
                    ]
                ),
            ):
                result = math_autobencher._generate_question_text_with_truth(
                    description,
                    "model",
                    None,
                    object(),
                    prefix,
                    question_count=1,
                    research_config=config,
                )
            self.assertEqual(result[0], [])
            failures = json.loads(
                Path(
                    f"{prefix}.generation_failures.json"
                ).read_text(encoding="utf-8")
            )
            self.assertTrue(
                any(
                    item.get("failure_type") == "difficulty_rejected"
                    for item in failures
                )
            )

    def test_sympy_gold_does_not_cross_compare_its_own_symbolic_answer(self):
        config, _ = load_project_config(
            ROOT / "configs" / "math_flywheel_smoke_test.yaml",
            temporary_overrides=["evaluator_pipeline.enabled=false"],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            prefix = str(Path(temp_dir, "subcat0"))
            with (
                patch.object(
                    math_autobencher,
                    "gen_from_prompt",
                    return_value=self._result(
                        [
                            {
                                "question": (
                                    "Solve for x: x^2 - 5*x + 6 = 0."
                                )
                            }
                        ]
                    ),
                ) as generate,
                patch.object(
                    math_autobencher,
                    "answers_equivalent",
                    side_effect=AssertionError(
                        "SymPy result must not be compared with itself"
                    ),
                ),
            ):
                result = (
                    math_autobencher._generate_question_text_with_truth(
                        self._description(
                            "Polynomials and Inequalities"
                        ),
                        "model",
                        None,
                        object(),
                        prefix,
                        question_count=1,
                        research_config=config,
                    )
                )

            self.assertEqual(generate.call_count, 1)
            self.assertEqual(result[0][0]["canonical_answer"], "{2, 3}")
            self.assertEqual(
                result[0][0]["truth_validation_details"]["solver_backend"],
                "sympy",
            )
            failures = json.loads(
                Path(
                    f"{prefix}.generation_failures.json"
                ).read_text(encoding="utf-8")
            )
            self.assertFalse(
                any(
                    item.get("failure_type")
                    == FailureType.TRUTH_DISAGREEMENT.value
                    for item in failures
                )
            )

    def test_leaked_partial_gold_adds_feedback_before_retry(self):
        config, _ = load_project_config(
            ROOT / "configs" / "math_flywheel_smoke_test.yaml",
            temporary_overrides=["evaluator_pipeline.enabled=false"],
        )
        faulty_question = (
            "Solve the system for (x, y, z): "
            "x + 2y - z = 5, 2x - y + 3z = 4, "
            "-x + 3y + 2z = 7."
        )
        invalid = [
            {
                "question": faulty_question,
                "candidate_gold_answer": "(2, 1, -1)",
            }
        ]
        valid = [{"question": "Solve for x: x + 4 = 9."}]
        with tempfile.TemporaryDirectory() as temp_dir:
            prefix = str(Path(temp_dir, "subcat0"))
            with patch.object(
                math_autobencher,
                "gen_from_prompt",
                side_effect=[
                    self._result(invalid),
                    self._result(valid),
                ],
            ) as generate:
                result = (
                    math_autobencher._generate_question_text_with_truth(
                        self._description("Systems of Equations"),
                        "model",
                        None,
                        object(),
                        prefix,
                        question_count=1,
                        research_config=config,
                    )
                )
            self.assertEqual(result[0][0]["canonical_answer"], "5")
            second_prompt = generate.call_args_list[1].kwargs["prompt"][0]
            self.assertIn("passed 1/3 equations", second_prompt)
            failures = json.loads(
                Path(
                    f"{prefix}.generation_failures.json"
                ).read_text(encoding="utf-8")
            )
            self.assertTrue(
                any(
                    item.get("failure_type")
                    == FailureType.PARTIAL_SOLUTION.value
                    for item in failures
                )
            )

    def test_truth_solver_failure_is_discarded_without_llm_repair(self):
        config, _ = load_project_config(
            ROOT / "configs" / "math_flywheel_smoke_test.yaml",
            temporary_overrides=["evaluator_pipeline.enabled=false"],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            prefix = str(Path(temp_dir, "subcat0"))
            with patch.object(
                math_autobencher,
                "gen_from_prompt",
                return_value=self._result(
                    [{"question": "Describe a beautiful number."}]
                ),
            ) as generate:
                result = (
                    math_autobencher._generate_question_text_with_truth(
                        self._description(),
                        "model",
                        None,
                        object(),
                        prefix,
                        question_count=1,
                        research_config=config,
                    )
                )
            self.assertEqual(result, [[]])
            self.assertEqual(generate.call_count, 1)
            summary = json.loads(
                Path(
                    f"{prefix}.generation_batch_summary.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                summary["failure_counts"]["truth_parse_fail"],
                1,
            )

    def test_invalid_decimal_gold_contract_rejects_only_candidate(self):
        config, _ = load_project_config(
            ROOT / "configs" / "math_flywheel_smoke_test.yaml",
            temporary_overrides=["evaluator_pipeline.enabled=false"],
        )
        invalid_truth = SimpleNamespace(
            success=True,
            canonical_answer="x + 1",
            answer_type="decimal",
            truth_validation_details={"substitution_passed": True},
        )
        solver = SimpleNamespace(
            solve=lambda _question: invalid_truth,
            training_reasoning=lambda _truth: [],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            prefix = str(Path(temp_dir, "invalid_decimal"))
            with (
                patch.object(
                    math_autobencher,
                    "gen_from_prompt",
                    return_value=self._result(
                        [{"question": "Compute the decimal value of 2 / 3."}]
                    ),
                ),
                patch.object(
                    math_autobencher.TruthSolver,
                    "from_config",
                    return_value=solver,
                ),
            ):
                result = math_autobencher._generate_question_text_with_truth(
                    self._description("Fraction and Decimal Operations"),
                    "model",
                    None,
                    object(),
                    prefix,
                    question_count=1,
                    research_config=config,
                )

            self.assertEqual(result, [[]])
            summary = json.loads(
                Path(f"{prefix}.generation_batch_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                summary["failure_counts"][FailureType.TRUTH_PARSE_FAIL.value],
                1,
            )

    def test_truth_solver_exception_rejects_only_candidate(self):
        config, _ = load_project_config(
            ROOT / "configs" / "math_flywheel_smoke_test.yaml",
            temporary_overrides=["evaluator_pipeline.enabled=false"],
        )
        solver = SimpleNamespace(
            solve=lambda _question: (_ for _ in ()).throw(
                ValueError("malformed candidate expression")
            )
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            prefix = str(Path(temp_dir, "solver_exception"))
            with (
                patch.object(
                    math_autobencher,
                    "gen_from_prompt",
                    return_value=self._result(
                        [{"question": "Compute the malformed expression."}]
                    ),
                ),
                patch.object(
                    math_autobencher.TruthSolver,
                    "from_config",
                    return_value=solver,
                ),
            ):
                result = math_autobencher._generate_question_text_with_truth(
                    self._description(),
                    "model",
                    None,
                    object(),
                    prefix,
                    question_count=1,
                    research_config=config,
                )

            self.assertEqual(result, [[]])
            summary = json.loads(
                Path(f"{prefix}.generation_batch_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                summary["failure_counts"][FailureType.TRUTH_PARSE_FAIL.value],
                1,
            )

    def test_invalid_candidate_is_skipped_while_valid_sibling_continues(self):
        config, _ = load_project_config(
            ROOT / "configs" / "math_flywheel_smoke_test.yaml",
            temporary_overrides=["evaluator_pipeline.enabled=false"],
        )
        mixed_batch = [
            {"question": "Solve for x: x + 4 = 9."},
            {
                "question": "Solve for x: x + 8 = 10.",
                "answer": "2",
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            prefix = str(Path(temp_dir, "subcat0"))
            with patch.object(
                math_autobencher,
                "gen_from_prompt",
                return_value=self._result(mixed_batch),
            ) as generate:
                result = math_autobencher._generate_question_text_with_truth(
                    self._description(),
                    "model",
                    None,
                    object(),
                    prefix,
                    question_count=2,
                    research_config=config,
                )
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(len(result[0]), 1)
        self.assertEqual(result[0][0]["canonical_answer"], "5")

    def test_gold_evaluator_parallelizes_independent_questions(self):
        config, _ = load_project_config(
            ROOT / "configs" / "math_flywheel_smoke_test.yaml",
            temporary_overrides=[
                "generation.gold_solver_backend=llm_python",
                "evaluator_pipeline.enabled=true",
                "evaluator_pipeline.max_parallel_questions=3",
            ],
        )
        questions = [
            {"question": "Solve for x: x + 4 = 9."},
            {"question": "Solve for x: x + 8 = 10."},
            {"question": "Solve for x: x + 2 = 5."},
        ]
        answers = {"9": "5", "10": "2", "5": "3"}
        state = {"active": 0, "maximum": 0}
        lock = threading.Lock()

        def solve(question, *_args, **_kwargs):
            with lock:
                state["active"] += 1
                state["maximum"] = max(
                    state["maximum"],
                    state["active"],
                )
            time.sleep(0.03)
            with lock:
                state["active"] -= 1
            right_hand_side = question.rsplit("=", 1)[1].strip(" .")
            answer = answers[right_hand_side]
            return {
                "status": "passed",
                "canonical_answer": answer,
                "answer_type": "integer",
                "verification_passed": True,
                "substitution_passed": True,
                "training_reasoning_summary": [
                    f"Rearrange the equation to obtain x = {answer}.",
                    f"Substitute x = {answer} into the original equation.",
                ],
                "estimated_difficulty": 2,
                "difficulty_acceptable": True,
            }

        api_client = SimpleNamespace(
            chat=SimpleNamespace(completions=object())
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            prefix = str(Path(temp_dir, "subcat0"))
            with (
                patch.object(
                    math_autobencher,
                    "gen_from_prompt",
                    return_value=self._result(questions),
                ),
                patch.object(
                    math_autobencher,
                    "solve_with_privileged_python",
                    side_effect=solve,
                ),
            ):
                result = (
                    math_autobencher._generate_question_text_with_truth(
                        self._description(),
                        "model",
                        None,
                        api_client,
                        prefix,
                        question_count=3,
                        research_config=config,
                    )
                )

        self.assertGreater(state["maximum"], 1)
        self.assertEqual(
            [item["canonical_answer"] for item in result[0]],
            ["5", "2", "3"],
        )


class InferenceResumeTests(unittest.TestCase):
    @staticmethod
    def _questions():
        return [
            {
                "id": str(index),
                "question": f"What is {index} + 1?",
                "answer": str(index + 1),
                "category": "Arithmetic",
            }
            for index in range(1, 4)
        ]

    def test_partial_cache_is_resumed_without_repeating_completed_items(self):
        questions = self._questions()
        cached = dict(questions[0], test_taker_response="2")
        completion = type("Completion", (), {})
        request_result = type("RequestResult", (), {})

        def result(text):
            item = completion()
            item.text = text
            wrapper = request_result()
            wrapper.completions = [item]
            return wrapper

        with tempfile.TemporaryDirectory() as temp_dir:
            prefix = str(Path(temp_dir, "iteration"))
            inference_file = Path(f"{prefix}.test_taker_inference.json")
            inference_file.write_text(
                json.dumps(cached),
                encoding="utf-8",
            )
            with patch.object(
                tool_util,
                "gen_from_prompt",
                side_effect=[result("3"), result("4")],
            ) as generate:
                answers = tool_util._generate_lm_answers(
                    questions,
                    ("model", None, object()),
                    None,
                    prefix,
                )

            self.assertEqual(generate.call_count, 2)
            self.assertEqual(len(answers), 3)
            saved = [
                json.loads(line)
                for line in inference_file.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(saved), 3)
            self.assertEqual(
                [line["question"] for line in saved],
                [line["question"] for line in questions],
            )

    def test_stale_cache_suffix_is_discarded(self):
        questions = self._questions()
        valid = dict(questions[0], test_taker_response="2")
        stale = dict(questions[1], question="different", test_taker_response="3")
        with tempfile.TemporaryDirectory() as temp_dir:
            inference_file = Path(temp_dir, "inference.json")
            inference_file.write_text(
                f"{json.dumps(valid)}\n{json.dumps(stale)}\n",
                encoding="utf-8",
            )
            loaded = tool_util._load_valid_inference_cache(
                str(inference_file),
                questions,
            )
            self.assertEqual(loaded, [valid])
            self.assertEqual(
                len(inference_file.read_text(encoding="utf-8").splitlines()),
                1,
            )


class MathJsonGovernanceTests(unittest.TestCase):
    def test_standard_inference_drives_stats_and_deduplicated_hard_pool(self):
        records = [
            tool_util.canonicalize_math_record(
                {
                    "id": index,
                    "category": "Algebra",
                    "sub_category": "Linear Equations",
                    "difficulty": 6,
                    "question": f"Solve x + {index} = 10.",
                    "answer": str(10 - index),
                    "test_taker_response": "0",
                    "is_correct": index == 1,
                    "error_tags": (
                        [] if index == 1 else ["concept_confusion"]
                    ),
                },
                index - 1,
            )
            for index in range(1, 4)
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            inference_file = Path(temp_dir, "run.test_taker_inference.json")
            hard_pool_file = Path(temp_dir, "hard_pool.json")
            dump_standard_json(records, inference_file)
            added, total = manage_hard_pool(
                inference_file, hard_pool_file, source_iter=1
            )
            self.assertEqual((added, total), (2, 2))
            self.assertEqual(
                manage_hard_pool(
                    inference_file, hard_pool_file, source_iter=1
                ),
                (0, 2),
            )
            hard_samples = json.loads(
                hard_pool_file.read_text(encoding="utf-8")
            )
            self.assertTrue(
                all(
                    set(sample["error_tags"]).issubset(ERROR_TAGS)
                    for sample in hard_samples
                )
            )
            summary = math_autobencher._build_compare_summary(1, records)
            self.assertEqual(summary["total_questions"], 3)
            self.assertEqual(len(summary["category_statistics"]), 1)
            self.assertIn("\n  {", inference_file.read_text(encoding="utf-8"))

    def test_cleanup_removes_fragments_and_preserves_core_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            iteration_dir = Path(temp_dir, "iter_1")
            temp_log = iteration_dir / "temp_log"
            temp_log.mkdir(parents=True)
            core = iteration_dir / "run.1.test_taker_inference.json"
            dump_standard_json([{"id": 1}], core)
            (iteration_dir / "run.1.subcat0.questions.json").write_text(
                "[]", encoding="utf-8"
            )
            (temp_log / "run.attempt1.txt").write_text(
                "invalid response", encoding="utf-8"
            )
            (iteration_dir / "broken.json").write_text("", encoding="utf-8")

            removed = clean_redundant_files(iteration_dir)

            self.assertEqual(len(removed), 3)
            self.assertTrue(core.exists())
            self.assertEqual(list(temp_log.iterdir()), [])

    def test_iteration_cleanup_helper_removes_failed_attempt_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = type(
                "Args",
                (),
                {
                    "clean_cycle_cache": True,
                    "mode": "data_flywheel",
                    "outfile_prefix1": str(Path(temp_dir, "run.")),
                    "research_run": None,
                },
            )()
            paths = math_autobencher._build_iteration_paths(
                args.outfile_prefix1,
                1,
                cycle_number=1,
            )
            attempt = Path(
                paths["temp_log_dir"],
                "generation.attempt1.txt",
            )
            attempt.write_text("invalid response", encoding="utf-8")
            audit = Path(
                paths["iteration_dir"],
                "run.gold_answer_validation.json",
            )
            dump_standard_json([{"status": "failed"}], audit)
            core_paths = {
                Path(paths["plan_file"]),
                Path(paths["inference_file"]),
                Path(paths["compare_file"]),
            }
            for core_path in core_paths:
                dump_standard_json([{"status": "core"}], core_path)
            subcat = Path(
                paths["iteration_dir"],
                "run.cycle1.iter1.subcat0.questions.json",
            )
            dump_standard_json([{"status": "temporary"}], subcat)

            removed = math_autobencher._cleanup_iteration_cache(
                args,
                1,
                1,
            )

            self.assertIn(str(attempt), removed)
            self.assertFalse(attempt.exists())
            self.assertFalse(audit.exists())
            self.assertFalse(subcat.exists())
            self.assertEqual(
                {
                    path
                    for path in Path(paths["iteration_dir"]).glob("*.json")
                },
                core_paths,
            )


if __name__ == "__main__":
    unittest.main()
