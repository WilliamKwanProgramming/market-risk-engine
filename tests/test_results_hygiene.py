"""No result file has a price, index-level or index-return column.

Derived series still determine the data they come from (docs/protocol.md, Part 6); this test
checks what the repository can check: the column names.
"""

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
BANNED = {"spot", "price", "prices", "close", "adj_close", "index", "index_level", "level"}


def test_no_result_file_has_a_price_or_index_level_column():
    for path in sorted((REPO / "results").rglob("*.csv")):
        columns = {c.lower() for c in pd.read_csv(path, nrows=0).columns}
        assert not columns & BANNED, f"{path.relative_to(REPO)} has {sorted(columns & BANNED)}"


def test_spx_forecasts_do_not_store_the_index_return():
    path = REPO / "results" / "realdata" / "forecasts_SPX.csv"
    if path.exists():
        columns = pd.read_csv(path, nrows=0).columns
        assert "realized" not in columns, "the SPX P&L is the index return, i.e. the input data"
