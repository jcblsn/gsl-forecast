import os
import sys

import duckdb
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def conn():
    with duckdb.connect(":memory:") as conn:
        yield conn
