"""Check that visible summary scores agree with the committed report."""
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("page", ["🏠 Overview", "💼 Recommendations"])
def test_dashboard_displays_recorded_metrics(page):
    recorded = json.loads((ROOT / "outputs/models/metrics.json").read_text())
    app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"))
    app.run(timeout=30)
    app.sidebar.radio[0].set_value(page).run(timeout=30)
    assert not app.exception
    visible = {metric.label: metric.value for metric in app.metric}
    assert visible["Recorded holdout RMSE"] == f"{recorded['models']['xgboost_tuned']['rmse']:.2f}"
    assert visible["Recorded holdout MAE"] == f"{recorded['models']['xgboost_tuned']['mae']:.2f}"
    assert visible["Recorded CV mean RMSE"] == f"{recorded['cross_validation']['mean_rmse']:.2f}"
    assert "Coverage" not in visible
