import duckdb
import pytest


@pytest.fixture
def conn():
    with duckdb.connect(":memory:") as conn:
        yield conn
