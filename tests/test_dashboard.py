"""Smoke test of app/dashboard.py: it renders without exception for every portfolio and method.

Skips when streamlit is not installed (pip install -e '.[dashboard]') or the precomputed
real-data results are missing (scripts/reproduce.sh realdata).
"""

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "app" / "dashboard.py"
PORTFOLIOS = ["EQ20", "SPX", "STRANGLE"]
METHODS = ["HS250", "HS500", "HS1000", "EWMA-N", "EWMA-T", "FHS", "MC"]

pytest.importorskip("streamlit", reason="pip install -e '.[dashboard]'")
if not (REPO / "results" / "realdata" / "summary.csv").exists():
    pytest.skip("results/realdata is missing: run scripts/reproduce.sh realdata",
                allow_module_level=True)

from streamlit.testing.v1 import AppTest  # noqa: E402


@pytest.mark.parametrize("portfolio", PORTFOLIOS)
def test_dashboard_renders_every_method(portfolio):
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    for method in METHODS:
        at.selectbox[0].set_value(portfolio)
        at.selectbox[1].set_value(method)
        at.run()
        assert not at.exception, (portfolio, method, at.exception)
    for radio in at.radio:  # every stress window, parameter set and Euler window
        for option in radio.options:
            radio.set_value(option)
            at.run()
            assert not at.exception, (portfolio, radio.label, option)
    assert "About this data" in [e.label for e in at.expander]
