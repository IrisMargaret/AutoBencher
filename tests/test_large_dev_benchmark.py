from collections import Counter

from autobencher.large_dev_benchmark import build_large_development_payload


def test_large_development_benchmark_is_static_balanced_and_not_blind():
    first = build_large_development_payload()
    second = build_large_development_payload()
    assert first == second
    assert first["role"] == "development_regression"
    assert "prohibited" in first["description"]
    assert len(first["questions"]) == 540
    counts = Counter(
        (item["category"], item["sub_category"])
        for item in first["questions"]
    )
    assert len(counts) == 27
    assert set(counts.values()) == {20}
    assert len({item["question"] for item in first["questions"]}) == 540
    assert all(
        item["source_dataset"] == "project_native_synthetic"
        for item in first["questions"]
    )
