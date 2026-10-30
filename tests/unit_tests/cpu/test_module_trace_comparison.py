import torch

from tests.integration_tests.compare_module_traces import compare_traces


def test_comparison_reports_missing_and_shape_mismatched_evidence():
    records = compare_traces(
        {"missing": torch.ones(2), "shape": torch.ones(2, 2)},
        {"shape": torch.ones(4)},
    )
    assert [r["status"] for r in records] == ["missing_tensor", "shape_mismatch"]
    assert all(not r["exact"] for r in records)


def test_comparison_reports_numeric_and_nonfinite_differences():
    records = compare_traces(
        {"finite": torch.tensor([1.0, 2.0]), "nan": torch.tensor([float("nan")])},
        {"finite": torch.tensor([1.0, 3.0]), "nan": torch.tensor([float("nan")])},
    )
    assert records[0]["max_ae"] == 1
    assert records[0]["different"] == 1
    assert not records[1]["finite"]
    assert not records[1]["exact"]
