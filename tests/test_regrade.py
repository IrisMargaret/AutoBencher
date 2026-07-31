import json
from pathlib import Path

from autobencher.config import load_resolved_config
from autobencher.regrade import regrade_run


def _write(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_regrade_is_offline_versioned_and_preserves_original(tmp_path):
    config = load_resolved_config(
        Path(__file__).parents[1] / "configs" / "math_flywheel_smoke_test.yaml"
    )[0]
    run_dir = tmp_path / "run"
    source = (
        run_dir
        / "fixed_test"
        / "baseline"
        / "fixed_math.test_taker_inference.json"
    )
    _write(run_dir / "resolved_config.json", config)
    rows = [
        {
            "question_id": "matrix-wrong",
            "answer_type": "matrix",
            "gold_answer": "[[5,4],[4,5]]",
            "test_taker_response": "[[5,6],[6,5]]",
            "is_correct": True,
        },
        {
            "question_id": "symbolic-right",
            "answer_type": "symbolic_expression",
            "gold_answer": "(x^2+2*x+1)*exp(x)",
            "test_taker_response": "exp(x)*(x^2+2x+1)",
            "is_correct": False,
        },
    ]
    _write(source, rows)
    original_bytes = source.read_bytes()

    result = regrade_run(run_dir)
    stage = result["stages"]["baseline"]
    assert stage["original_correct"] == 1
    assert stage["regraded_correct"] == 1
    assert stage["correct_to_wrong"] == ["matrix-wrong"]
    assert stage["wrong_to_correct"] == ["symbolic-right"]
    assert source.read_bytes() == original_bytes
    assert Path(stage["output_path"]).is_file()

    # Re-running the same evaluator version is idempotent, not destructive.
    assert regrade_run(run_dir)["stages"]["baseline"] == stage
