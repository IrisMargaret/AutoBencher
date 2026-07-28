import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import math_autobencher
import tool_util
from autobencher.config import load_project_config
from tool_util import (
    ERROR_TAGS,
    clean_redundant_files,
    dump_standard_json,
    extract_json_v2,
    manage_hard_pool,
)

ROOT = Path(__file__).resolve().parents[1]


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


if __name__ == "__main__":
    unittest.main()
