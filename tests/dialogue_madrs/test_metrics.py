from dyadic_sim.experiments.dialogue_madrs.metrics import (
    evaluation_report,
    regression_metrics,
)


def test_regression_metrics_include_required_measures():
    metrics = regression_metrics([0, 2, 4], [0.2, 2.6, 5.8])

    assert set(metrics) == {
        "count",
        "mae",
        "rmse",
        "pearson",
        "exact_rounded_accuracy",
        "within_1_accuracy",
    }
    assert metrics["exact_rounded_accuracy"] == 1 / 3
    assert metrics["within_1_accuracy"] == 2 / 3


def test_evaluation_report_groups_by_symptom():
    rows = [
        {"symptom": "Sleep", "severity_score": 1, "prediction": 1.2},
        {"symptom": "Sleep", "severity_score": 3, "prediction": 2.8},
        {"symptom": "Sadness", "severity_score": 2, "prediction": 2.1},
    ]

    report = evaluation_report(rows)

    assert report["overall"]["count"] == 3
    assert report["per_symptom"]["Sleep"]["count"] == 2
    assert report["per_symptom"]["Sadness"]["pearson"] is None
