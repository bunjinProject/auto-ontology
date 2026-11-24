# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dremio connector tests.

Driver-free: the connector runs against a fake ADBC connection, so no Dremio
cluster has to be reachable.
"""

from __future__ import annotations

import io
import json
import time
import urllib.error
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pyarrow as pa
import pytest

from auto_ontology.connectors.base import StatementTimeout
from auto_ontology.connectors.connection_string_factory import build_connection_string
from auto_ontology.connectors.dremio import (
    DremioDatabase,
    _parse_connection_string,
    _qualified,
)
from auto_ontology.connectors.registry import CONNECTOR_REGISTRY, create_connector

HOST = "dremio.example.com"


def _connection(**overrides: object) -> dict[str, object]:
    connection: dict[str, object] = {
        "type": "dremio",
        "host": HOST,
        "password": "PAT123",
        "database": "lake",
    }
    connection.update(overrides)
    return connection


def _database(**overrides: object) -> DremioDatabase:
    return DremioDatabase(build_connection_string(_connection(**overrides)))


class _FakeStatement:
    def __init__(self, recorder: "_FakeConnection") -> None:
        self._recorder = recorder

    def set_options(self, **options: str) -> None:
        self._recorder.options.update(options)


class _FakeCursor:
    def __init__(self, recorder: "_FakeConnection") -> None:
        self._recorder = recorder
        self.adbc_statement = _FakeStatement(recorder)
        self.description: list[tuple[str, str]] | None = None

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, sql: str, parameters: Any = None) -> None:
        self._recorder.statements.append(sql)
        if self._recorder.error is not None:
            raise self._recorder.error
        self.description = self._recorder.description

    def fetch_arrow_table(self) -> pa.Table:
        return pa.Table.from_pandas(self._recorder.frame)


class _FakeConnection:
    def __init__(self) -> None:
        self.statements: list[str] = []
        self.options: dict[str, str] = {}
        self.frame = pd.DataFrame()
        self.description: list[tuple[str, str]] | None = [("c", "x")]
        self.error: Exception | None = None
        self.closed = False

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self)

    def close(self) -> None:
        self.closed = True


def _wire(database: DremioDatabase) -> _FakeConnection:
    fake = _FakeConnection()
    database._connection = fake  # noqa: SLF001 - test double for the driver
    return fake


# ----------------------------------------------------------------------
# Connection string
# ----------------------------------------------------------------------


def test_dremio_is_registered() -> None:
    assert CONNECTOR_REGISTRY["dremio"] is DremioDatabase


def test_create_connector_builds_a_dremio_database() -> None:
    connector = create_connector(build_connection_string(_connection()))

    assert isinstance(connector, DremioDatabase)


def test_default_port_is_the_flight_port() -> None:
    assert urlparse(build_connection_string(_connection())).port == 32010


def test_token_and_name_round_trip() -> None:
    settings = _parse_connection_string(
        build_connection_string(_connection(password="p@ss/word"))
    )

    assert settings["token"] == "p@ss/word"
    assert settings["name"] == "lake"
    assert settings["tls"] is False


def test_tls_flags_are_carried_through() -> None:
    settings = _parse_connection_string(
        build_connection_string(_connection(tls=True, tls_skip_verify=True))
    )

    assert settings["tls"] is True
    assert settings["tls_skip_verify"] is True


def test_a_token_is_required() -> None:
    with pytest.raises(ValueError):
        build_connection_string(_connection(password=""))
    with pytest.raises(ValueError):
        _parse_connection_string(f"dremio://user@{HOST}:32010/lake")


def test_name_defaults_when_the_url_has_no_path() -> None:
    assert _parse_connection_string(f"dremio://t:x@{HOST}")["name"] == "dremio"


def test_wrong_scheme_is_rejected() -> None:
    with pytest.raises(ValueError):
        _parse_connection_string(f"trino://t:x@{HOST}/hive")


def test_pinned_schema_seeds_the_allowlist() -> None:
    database = _database(schema="sales")

    assert database._schema_names == ["sales"]  # noqa: SLF001


# ----------------------------------------------------------------------
# Identity
# ----------------------------------------------------------------------


def test_dialect_and_database_name() -> None:
    database = _database()

    assert database.dialect == "dremio"
    assert database.database_name == "lake"


def test_qualify_quotes_every_path_segment() -> None:
    database = _database()

    assert database.qualify("my source.folder", "order") == (
        '"my source"."folder"."order"'
    )
    assert _qualified(None, "t") == '"t"'
    assert _qualified('a"b', "t") == '"a""b"."t"'


# ----------------------------------------------------------------------
# Execution and introspection
# ----------------------------------------------------------------------


def test_execute_returns_a_dataframe() -> None:
    database = _database()
    fake = _wire(database)
    fake.frame = pd.DataFrame({"a": [1, 2]})

    result = database.execute("SELECT 1")

    assert result["a"].tolist() == [1, 2]
    assert fake.statements == ["SELECT 1"]


def test_statement_without_a_result_set_is_an_empty_frame() -> None:
    database = _database()
    fake = _wire(database)
    fake.description = None

    assert database.execute("DROP VIEW x").empty


def test_timeout_sets_the_rpc_deadline_for_that_statement() -> None:
    database = _database()
    fake = _wire(database)

    database.execute("SELECT 1", timeout_s=7)

    assert fake.options["adbc.flight.sql.rpc.timeout_seconds.query"] == "7.0"


def test_deadline_exceeded_becomes_a_statement_timeout() -> None:
    database = _database()
    fake = _wire(database)
    fake.error = RuntimeError("Deadline Exceeded")

    with pytest.raises(StatementTimeout):
        database.execute("SELECT 1", timeout_s=5)


def test_other_errors_propagate_unchanged() -> None:
    database = _database()
    fake = _wire(database)
    fake.error = RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        database.execute("SELECT 1", timeout_s=5)


def test_get_tables_maps_table_types_and_drops_system_schemas() -> None:
    database = _database()
    fake = _wire(database)
    fake.frame = pd.DataFrame(
        {
            "table_schema": ["sales", "sales.2024"],
            "table_name": ["orders", "v"],
            "table_type": ["base table", "view"],
        }
    )

    tables = database.get_tables()

    assert tables["table_name"].tolist() == ["orders", "v"]
    sql = fake.statements[0]
    assert "SYSTEM_TABLE" in sql
    assert "'information_schema'" in sql


def test_allowlist_covers_nested_folders_only() -> None:
    database = DremioDatabase(build_connection_string(_connection()), schemas=["Sales"])
    fake = _wire(database)
    fake.frame = pd.DataFrame(
        {
            "table_schema": ["sales", "sales.2024", "salesforce", "other"],
            "table_name": ["a", "b", "c", "d"],
            "table_type": ["base table"] * 4,
        }
    )

    assert database.get_tables()["table_name"].tolist() == ["a", "b"]


def test_get_schemas_returns_top_level_roots_once() -> None:
    database = _database()
    fake = _wire(database)
    fake.frame = pd.DataFrame({"SCHEMA_NAME": ["sales", "sales.2024", "hr"]})

    assert database.get_schemas() == ["hr", "sales"]


def test_get_columns_selects_a_boolean_nullable_flag() -> None:
    database = _database()
    fake = _wire(database)
    fake.frame = pd.DataFrame(
        {
            "table_schema": ["sales"],
            "table_name": ["orders"],
            "column_name": ["id"],
            "data_type": ["INTEGER"],
            "is_nullable": [False],
            "ordinal_position": [1],
        }
    )

    columns = database.get_columns()

    assert columns["is_nullable"].tolist() == [False]
    assert "IS_NULLABLE = 'YES'" in fake.statements[0]


def test_get_queries_is_empty_when_history_is_unreadable() -> None:
    database = _database()
    fake = _wire(database)
    fake.error = RuntimeError("permission denied")

    result = database.get_queries()

    assert result.empty
    assert list(result.columns) == ["end_time", "query_text"]


def test_keys_are_empty_but_shaped() -> None:
    database = _database()

    assert database.get_pks().empty
    assert "referenced_column" in database.get_fks().columns


def test_close_releases_the_connection() -> None:
    database = _database()
    fake = _wire(database)

    database.close()

    assert fake.closed
    assert database._connection is None  # noqa: SLF001


# ----------------------------------------------------------------------
# Token exchange
# ----------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode()


def _http_error(code: int, body: str = "") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://x",
        code,
        "err",
        {},
        io.BytesIO(body.encode()),  # type: ignore[arg-type]
    )


def _patch_exchange(monkeypatch: pytest.MonkeyPatch, result: Any) -> list[Any]:
    """Replace ``urlopen``; *result* is a payload dict or an exception to raise."""
    requests: list[Any] = []

    def fake_urlopen(request: Any, timeout: float, context: Any) -> _FakeResponse:
        requests.append((request, context))
        if isinstance(result, Exception):
            raise result
        return _FakeResponse(result)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return requests


def test_rest_port_defaults_and_can_be_overridden() -> None:
    assert _parse_connection_string(f"dremio://t:x@{HOST}")["rest_port"] == 9047
    assert (
        _parse_connection_string(f"dremio://t:x@{HOST}?rest_port=9443")["rest_port"]
        == 9443
    )
    with pytest.raises(ValueError):
        _parse_connection_string(f"dremio://t:x@{HOST}?rest_port=abc")


def test_exchange_sends_the_pat_and_returns_an_expiring_access_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = _patch_exchange(
        monkeypatch, {"access_token": "OAUTH", "expires_in": 3600}
    )
    database = _database(tls=True, tls_skip_verify=True)

    token, expires_at = database._exchange_token()  # noqa: SLF001

    request, context = requests[0]
    assert token == "OAUTH"
    assert expires_at is not None and expires_at > time.monotonic()
    assert request.full_url == f"https://{HOST}:9047/oauth/token"
    body = parse_qs(request.data.decode())
    assert body["subject_token"] == ["PAT123"]
    assert body["scope"] == ["dremio.all"]
    assert context is not None  # certificate verification is off


def test_exchange_uses_plain_http_without_tls(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = _patch_exchange(monkeypatch, {"access_token": "T", "expires_in": 60})

    _database()._exchange_token()  # noqa: SLF001

    assert requests[0][0].full_url.startswith("http://")
    assert requests[0][1] is None


def test_missing_exchange_route_falls_back_to_the_pat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_exchange(monkeypatch, _http_error(404))

    assert _database()._exchange_token() == ("PAT123", None)  # noqa: SLF001


def test_a_rejected_pat_is_reported_not_papered_over(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_exchange(monkeypatch, _http_error(400, "invalid_grant"))

    with pytest.raises(ConnectionError, match="HTTP 400"):
        _database()._exchange_token()  # noqa: SLF001


def test_an_unreachable_rest_api_names_the_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_exchange(monkeypatch, urllib.error.URLError("refused"))

    with pytest.raises(ConnectionError, match="9047"):
        _database()._exchange_token()  # noqa: SLF001


def test_the_access_token_not_the_pat_goes_to_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_exchange(monkeypatch, {"access_token": "OAUTH", "expires_in": 3600})
    seen: dict[str, Any] = {}

    def fake_connect(uri: str, db_kwargs: dict[str, str]) -> _FakeConnection:
        seen.update(uri=uri, db_kwargs=db_kwargs)
        return _FakeConnection()

    monkeypatch.setattr("adbc_driver_flightsql.dbapi.connect", fake_connect)

    _database(tls=True, tls_skip_verify=True)._open()  # noqa: SLF001

    assert seen["uri"] == f"grpc+tls://{HOST}:32010"
    assert seen["db_kwargs"]["adbc.flight.sql.authorization_header"] == "Bearer OAUTH"
    assert seen["db_kwargs"]["adbc.flight.sql.client_option.tls_skip_verify"] == "true"


def test_an_expired_access_token_reopens_the_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = _database()
    stale = _wire(database)
    database._token_expires_at = time.monotonic() - 1  # noqa: SLF001
    fresh = _FakeConnection()
    fresh.frame = pd.DataFrame({"a": [1]})
    monkeypatch.setattr(database, "_open", lambda: fresh)

    result = database.execute("SELECT 1")

    assert stale.closed
    assert result["a"].tolist() == [1]


def test_unauthenticated_is_retried_once_with_a_new_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = _database()
    rejecting = _wire(database)
    rejecting.error = RuntimeError("UNAUTHENTICATED: token expired")
    database._token_expires_at = time.monotonic() + 3600  # noqa: SLF001
    fresh = _FakeConnection()
    fresh.frame = pd.DataFrame({"a": [2]})
    monkeypatch.setattr(database, "_open", lambda: fresh)

    assert database.execute("SELECT 1")["a"].tolist() == [2]
    assert rejecting.closed


def test_unauthenticated_with_the_pat_as_bearer_is_not_retried() -> None:
    database = _database()
    fake = _wire(database)
    fake.error = RuntimeError("UNAUTHENTICATED")

    with pytest.raises(RuntimeError, match="UNAUTHENTICATED"):
        database.execute("SELECT 1")

    assert len(fake.statements) == 1


def test_allowlist_is_pushed_into_every_introspection_query() -> None:
    """A big Dremio holds hundreds of thousands of relations; read only the allowlist."""
    database = DremioDatabase(
        build_connection_string(_connection()), schemas=["Sales", "o'brien"]
    )
    fake = _wire(database)
    fake.frame = pd.DataFrame()

    database.get_tables()
    database.get_columns()
    database.get_views()

    assert len(fake.statements) == 3
    for sql in fake.statements:
        assert "LOWER(TABLE_SCHEMA) = 'sales'" in sql
        assert "LOWER(TABLE_SCHEMA) LIKE 'sales.%'" in sql
        assert "o''brien" in sql  # quote escaped, not injectable


def test_without_an_allowlist_nothing_extra_is_pushed_down() -> None:
    database = _database()
    fake = _wire(database)
    fake.frame = pd.DataFrame()

    database.get_tables()

    assert "LIKE 'sales" not in fake.statements[0]
    assert "OR LOWER" not in fake.statements[0]


def test_query_history_is_limited_to_the_allowlist() -> None:
    """sys.jobs_recent is cluster-wide; only queries on the chosen roots matter."""
    database = DremioDatabase(
        build_connection_string(_connection()), schemas=["Sales", "o'brien"]
    )
    fake = _wire(database)
    fake.frame = pd.DataFrame()

    database.get_queries(hours=6)

    sql = fake.statements[0]
    assert "LOWER(query) LIKE '%sales%'" in sql
    assert "o''brien" in sql
    assert "NOT LIKE 'alter %'" in sql
    assert "-6" in sql


def test_query_history_is_unscoped_without_an_allowlist() -> None:
    database = _database()
    fake = _wire(database)
    fake.frame = pd.DataFrame()

    database.get_queries()

    assert "LIKE '%" not in fake.statements[0]
