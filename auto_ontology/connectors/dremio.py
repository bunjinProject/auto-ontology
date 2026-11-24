# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dremio connector implementing Auto Ontology's SQLDatabase ABC.

Dremio is reached over Arrow Flight SQL (default port 32010) with the ADBC
driver, which returns Arrow tables that convert to DataFrames without a
row-by-row copy.

Three properties of the engine shape this connector:

* **Dotted namespace.** Dremio has no database level. Objects live under
  sources and spaces, nested through folders -- ``source.folder.table`` -- and
  ``INFORMATION_SCHEMA`` reports the whole path before the table name as one
  dotted ``TABLE_SCHEMA`` (``"my source.folder"``). :meth:`DremioDatabase.qualify`
  splits that path and quotes each segment. A folder whose own name contains a
  dot is therefore ambiguous; Dremio does not expose the segments separately.
* **Connection identity.** With no database to bind, the URL path is only the
  name NeMo routes SQL by (:attr:`~DremioDatabase.database_name`), and defaults
  to ``dremio``. Use one connection per cluster, or a distinct name per
  connection to the same cluster.
* **No constraints.** Dremio declares neither primary nor foreign keys, so
  :meth:`DremioDatabase.get_pks` and :meth:`DremioDatabase.get_fks` return
  correctly shaped *empty* frames.

Authentication is a personal access token (PAT). The username in the URL is a
label only. A self-hosted Dremio's Flight listener does not accept a PAT as a
bearer token (it answers ``UNAUTHENTICATED``), so the connector first trades it
at the REST API's ``/oauth/token`` for a short-lived access token (one hour) and
sends *that*. The access token is renewed shortly before it expires, and again
if the server rejects it. An endpoint that has no such route (HTTP 404/405) gets
the PAT as the bearer token directly.

``tls``
    ``true`` selects ``grpc+tls://`` for Flight and ``https`` for the token
    exchange. Defaults to false, which is Dremio's out-of-the-box listener.
``tls_skip_verify``
    Accept a certificate the container does not trust, such as one issued by an
    internal CA. Applies to both Flight and the token exchange. Defaults to
    false.
``rest_port``
    Port of the REST API used for the token exchange. Defaults to 9047.

Example
-------
::

    CONNECTION_STRINGS=dremio://token:PAT@dremio.example.com:32010/dremio
    CONNECTION_STRINGS=dremio://token:PAT@dremio.example.com:32010/dremio?tls=true
"""

from __future__ import annotations

import json
import logging
import ssl
import threading
import time
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import parse_qs, unquote, urlencode, urlparse

import pandas as pd
from auto_ontology.catalog.constants import TableTypes
from auto_ontology.connectors.base import SQLDatabase, StatementTimeout

if TYPE_CHECKING:
    from adbc_driver_manager.dbapi import Connection

logger = logging.getLogger(__name__)

DEFAULT_PORT = 32010
DEFAULT_REST_PORT = 9047
DEFAULT_NAME = "dremio"

# OAuth 2.0 token exchange (RFC 8693) of a PAT for a Flight-usable access token.
_TOKEN_PATH = "/oauth/token"
_TOKEN_EXCHANGE_TIMEOUT_S = 15
# Renew this long before the access token expires, so a statement never starts
# with a token that dies mid-flight.
_TOKEN_REFRESH_MARGIN_S = 60
# Routes a server without token exchange answers with these; any other error is
# a real failure (bad PAT, wrong scope) and must not be papered over.
_NO_EXCHANGE_STATUS = (404, 405)

# Set on the DatabaseOptions / StatementOptions of adbc_driver_flightsql. Spelled
# out so importing this module never needs the driver.
_AUTHORIZATION_HEADER = "adbc.flight.sql.authorization_header"
_TLS_SKIP_VERIFY = "adbc.flight.sql.client_option.tls_skip_verify"
_TIMEOUT_QUERY = "adbc.flight.sql.rpc.timeout_seconds.query"
_TIMEOUT_FETCH = "adbc.flight.sql.rpc.timeout_seconds.fetch"

# Dremio's own metadata, never worth ingesting. Compared case-insensitively.
_SYSTEM_SCHEMAS = ("information_schema", "sys")

_TABLE_SCHEMA = ["table_schema", "table_name", "table_type"]

_COLUMN_SCHEMA = [
    "table_schema",
    "table_name",
    "column_name",
    "data_type",
    "is_nullable",
    "ordinal_position",
]

_VIEW_SCHEMA = ["table_schema", "table_name", "view_definition"]

_PK_SCHEMA = ["table_schema", "table_name", "column_name", "ordinal_position"]

_FK_SCHEMA = [
    "table_schema",
    "table_name",
    "column_name",
    "referenced_schema",
    "referenced_table",
    "referenced_column",
]

# Needs a role that can read sys.jobs_recent, so it is attempted and tolerated.
_QUERY_SCHEMA = ["end_time", "query_text"]


def _quoted_identifier(name: str) -> str:
    """Return an ANSI double-quoted identifier.

    Dremio source and folder names routinely contain spaces, dashes and
    mixed case, none of which survive unquoted.
    """
    return '"' + name.replace('"', '""') + '"'


def _qualified(schema_path: str | None, table: str) -> str:
    """Return a quoted reference for a dotted *schema_path* and *table*."""
    parts = [p for p in (schema_path or "").split(".") if p]
    parts.append(table)
    return ".".join(_quoted_identifier(part) for part in parts)


def _is_true(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _parse_connection_string(connection_string: str) -> dict[str, Any]:
    """Parse a Dremio URL into the settings this connector needs.

    Expected format::

        dremio://USER:PAT@HOST[:32010][/NAME][?tls=true&tls_skip_verify=true&schema=...]

    The password is the personal access token. The username is accepted for
    URL shape only and may be anything.
    """
    parsed = urlparse(connection_string)
    if parsed.scheme.split("+", 1)[0].lower() != "dremio":
        raise ValueError("Not a Dremio URL")
    if not parsed.hostname:
        raise ValueError("Invalid Dremio connection string (missing host)")

    token = unquote(parsed.password) if parsed.password else ""
    if not token:
        raise ValueError(
            "Dremio connection string requires a personal access token as the "
            "password, e.g. dremio://token:PAT@dremio.example.com:32010/dremio"
        )

    query = parse_qs(parsed.query)

    def first(key: str) -> str | None:
        value = query.get(key, [None])[0]
        return unquote(value) if value else None

    rest_port = first("rest_port")
    if rest_port is not None and not rest_port.isdigit():
        raise ValueError(f"Invalid Dremio rest_port: {rest_port!r}")

    return {
        "host": parsed.hostname,
        "port": parsed.port or DEFAULT_PORT,
        "rest_port": int(rest_port) if rest_port else DEFAULT_REST_PORT,
        "token": token,
        "name": unquote(parsed.path.lstrip("/")) or DEFAULT_NAME,
        "schema": first("schema"),
        "tls": _is_true(first("tls")),
        "tls_skip_verify": _is_true(first("tls_skip_verify")),
    }


def _under(path: str, roots: set[str]) -> bool:
    """Whether dotted *path* is one of *roots* or nested beneath one."""
    folded = path.casefold()
    return any(folded == root or folded.startswith(root + ".") for root in roots)


class DremioDatabase(SQLDatabase):
    """Concrete :class:`SQLDatabase` backed by Dremio over Arrow Flight SQL.

    Parameters
    ----------
    connection_string:
        ``dremio://user:PAT@host:32010/name[?tls=true]``
    schemas:
        Optional ingestion allowlist of sources, spaces or folder paths. An
        entry covers everything nested under it: ``sales`` matches ``sales`` and
        ``sales.2024``. Empty or ``None`` means everything except Dremio's
        system schemas. A ``?schema=`` in the URL seeds the same allowlist.
    """

    supports_statement_timeout = True

    def __init__(
        self,
        connection_string: str,
        schemas: list[str] | None = None,
    ) -> None:
        self._settings = _parse_connection_string(connection_string)
        self._name = str(self._settings["name"])

        allowlist = [s for s in (schemas or []) if s and s.strip()]
        pinned = self._settings["schema"]
        if not allowlist and pinned:
            allowlist = [str(pinned)]
        self._schema_names: list[str] = sorted({s.strip() for s in allowlist})
        self._schema_roots: set[str] | None = (
            {s.casefold() for s in self._schema_names} if self._schema_names else None
        )
        if self._schema_roots:
            logger.info(
                "Dremio ingestion restricted to schemas: %s", sorted(self._schema_roots)
            )

        self._lock = threading.RLock()
        self._connection: "Connection | None" = None
        # ``time.monotonic()`` deadline of the access token behind the open
        # connection; ``None`` when the PAT is used directly and never expires.
        self._token_expires_at: float | None = None

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    def _exchange_token(self) -> tuple[str, float | None]:
        """Trade the PAT for an access token the Flight listener accepts.

        Returns the bearer token and its ``time.monotonic()`` expiry (``None``
        when it does not expire). Raises on anything but a missing exchange
        route, so a revoked PAT surfaces as itself instead of as the opaque
        ``UNAUTHENTICATED`` the Flight listener would later give.
        """
        settings = self._settings
        scheme = "https" if settings["tls"] else "http"
        url = f"{scheme}://{settings['host']}:{settings['rest_port']}{_TOKEN_PATH}"
        body = urlencode(
            {
                "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
                "subject_token": settings["token"],
                "subject_token_type": (
                    "urn:ietf:params:oauth:token-type:dremio:personal-access-token"
                ),
                "requested_token_type": "urn:ietf:params:oauth:token-type:access_token",
                "scope": "dremio.all",
            }
        ).encode()
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        context = None
        if scheme == "https" and settings["tls_skip_verify"]:
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        try:
            with urllib.request.urlopen(  # noqa: S310 - scheme is fixed above
                request, timeout=_TOKEN_EXCHANGE_TIMEOUT_S, context=context
            ) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code in _NO_EXCHANGE_STATUS:
                logger.info(
                    "Dremio has no %s; using the PAT as the bearer", _TOKEN_PATH
                )
                return str(settings["token"]), None
            detail = exc.read()[:200].decode(errors="replace")
            raise ConnectionError(
                f"Dremio rejected the personal access token exchange "
                f"(HTTP {exc.code}): {detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, ssl.SSLError) as exc:
            raise ConnectionError(
                f"Could not reach Dremio's REST API at {scheme}://{settings['host']}:"
                f"{settings['rest_port']} for the token exchange: {exc}. Set "
                "rest_port if it listens elsewhere."
            ) from exc

        access_token = payload.get("access_token")
        if not access_token:
            raise ConnectionError("Dremio token exchange returned no access_token")
        expires_in = payload.get("expires_in")
        expires_at = (
            time.monotonic() + float(expires_in) - _TOKEN_REFRESH_MARGIN_S
            if expires_in
            else None
        )
        return str(access_token), expires_at

    def _open(self) -> "Connection":
        """Open an ADBC Flight SQL connection.

        The driver is imported lazily so importing this module stays cheap and a
        deployment that never configures Dremio is unaffected.
        """
        from adbc_driver_flightsql import dbapi

        bearer, self._token_expires_at = self._exchange_token()
        scheme = "grpc+tls" if self._settings["tls"] else "grpc"
        uri = f"{scheme}://{self._settings['host']}:{self._settings['port']}"
        db_kwargs = {_AUTHORIZATION_HEADER: f"Bearer {bearer}"}
        if self._settings["tls_skip_verify"]:
            db_kwargs[_TLS_SKIP_VERIFY] = "true"
        connection = dbapi.connect(uri, db_kwargs=db_kwargs)
        logger.info("Opened Dremio connection to %s", uri)
        return connection

    def _reset_connection(self) -> None:
        if self._connection is not None:
            try:
                self._connection.close()
            except Exception:
                logger.debug("Failed to close Dremio connection", exc_info=True)
            self._connection = None

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def dialect(self) -> str:
        """Return this engine's sqlglot dialect name.

        Must be a member of ``sqlglot.dialects.DIALECTS`` — callers pass it
        straight to sqlglot without translation. See ``CONNECTOR_REGISTRY``.
        """
        return "dremio"

    @property
    def database_name(self) -> str:
        return self._name

    def qualify(self, schema: Optional[str], table: str) -> str:
        """Quote each segment of the dotted schema path and the table name."""
        return _qualified(schema, table)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(
        self,
        sql: str,
        parameters: Optional[list] = None,
        *,
        timeout_s: float | None = None,
    ) -> pd.DataFrame:
        """Run *sql* and return the result as a DataFrame.

        A statement that returns no rows yields an empty frame. With *timeout_s*
        the RPC deadline is set on this statement only, so it cannot outlive it.
        """
        with self._lock:
            started = time.monotonic()
            # Renew before the access token lapses rather than failing a statement.
            if self._token_expires_at is not None and started >= self._token_expires_at:
                self._reset_connection()
            if self._connection is None:
                self._connection = self._open()
            try:
                try:
                    return self._run(sql, parameters, timeout_s)
                except Exception as exc:
                    # A server-side revocation or clock skew can kill the token
                    # early. Once, with a fresh one -- a second failure is real.
                    if self._token_expires_at is None or (
                        "unauthenticated" not in str(exc).casefold()
                    ):
                        raise
                    logger.info("Dremio rejected the access token; renewing it")
                    self._reset_connection()
                    self._connection = self._open()
                    return self._run(sql, parameters, timeout_s)
            except Exception as exc:
                if timeout_s is not None and "deadline" in str(exc).casefold():
                    raise StatementTimeout.since(timeout_s, started) from exc
                raise

    def _run(
        self, sql: str, parameters: Optional[list], timeout_s: float | None
    ) -> pd.DataFrame:
        """Execute one statement on the open connection."""
        assert self._connection is not None  # noqa: S101 - guarded by execute()
        with self._connection.cursor() as cursor:
            if timeout_s is not None:
                seconds = str(float(max(1, int(timeout_s))))
                cursor.adbc_statement.set_options(
                    **{_TIMEOUT_QUERY: seconds, _TIMEOUT_FETCH: seconds}
                )
            cursor.execute(sql, parameters or None)
            if cursor.description is None:
                return pd.DataFrame()
            return cursor.fetch_arrow_table().to_pandas()

    # ------------------------------------------------------------------
    # Schema introspection
    # ------------------------------------------------------------------

    @staticmethod
    def _system_condition(column: str = "TABLE_SCHEMA") -> str:
        """Exclude Dremio's own metadata schemas (and anything nested in them)."""
        clauses = [
            f"LOWER({column}) <> '{name}' AND LOWER({column}) NOT LIKE '{name}.%'"
            for name in _SYSTEM_SCHEMAS
        ]
        return " AND ".join(clauses)

    def _scope_condition(self, column: str = "TABLE_SCHEMA") -> str:
        """The ``WHERE`` condition selecting which schemas to introspect.

        The allowlist is pushed to the server. ``INFORMATION_SCHEMA`` on a large
        Dremio holds hundreds of thousands of relations, so filtering after the
        fact would still read all of them (minutes, and gigabytes of columns).

        ``LIKE`` treats ``_`` and ``%`` as wildcards, so a root such as ``a_b``
        can also match ``axb.t``. That only over-selects; :meth:`_filter_by_schema`
        then applies the exact test, so the result is still correct.
        """
        system = self._system_condition(column)
        if not self._schema_roots:
            return system
        roots = []
        for root in sorted(self._schema_roots):
            literal = root.replace("'", "''")
            roots.append(
                f"LOWER({column}) = '{literal}' OR LOWER({column}) LIKE '{literal}.%'"
            )
        return f"{system} AND ({' OR '.join(roots)})"

    def _filter_by_schema(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Keep rows under an allowlisted root; the roots match case-insensitively.

        The exact test behind :meth:`_scope_condition`, whose ``LIKE`` can
        over-select when a root contains ``_`` or ``%``.
        """
        if (
            self._schema_roots is None
            or frame.empty
            or "table_schema" not in frame.columns
        ):
            return frame
        keep = (
            frame["table_schema"]
            .astype(str)
            .map(lambda path: _under(path, self._schema_roots or set()))
        )
        return frame.loc[keep].reset_index(drop=True)

    def get_schemas(self) -> list[str]:
        """List top-level sources and spaces.

        Used by the connection UI to pick what to ingest, and by the connection
        test -- it runs a real query, so it proves the endpoint, the token and
        the listener all work.
        """
        frame = self.execute(
            'SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA."SCHEMATA" '
            f"WHERE {self._system_condition('SCHEMA_NAME')} ORDER BY SCHEMA_NAME"
        )
        if frame.empty:
            return []
        roots = {str(name).split(".", 1)[0] for name in frame.iloc[:, 0].tolist()}
        return sorted(roots, key=str.casefold)

    def get_tables(self) -> pd.DataFrame:
        frame = self.execute(f"""
            SELECT
                TABLE_SCHEMA AS table_schema,
                TABLE_NAME AS table_name,
                CASE TABLE_TYPE
                    WHEN 'VIEW' THEN '{TableTypes.VIEW}'
                    ELSE '{TableTypes.BASE_TABLE}'
                END AS table_type
            FROM INFORMATION_SCHEMA."TABLES"
            WHERE TABLE_TYPE <> 'SYSTEM_TABLE'
              AND {self._scope_condition()}
            ORDER BY TABLE_SCHEMA, TABLE_NAME
        """)
        if frame.empty:
            return pd.DataFrame(columns=_TABLE_SCHEMA)
        return self._filter_by_schema(frame)

    def get_columns(self) -> pd.DataFrame:
        frame = self.execute(f"""
            SELECT
                TABLE_SCHEMA AS table_schema,
                TABLE_NAME AS table_name,
                COLUMN_NAME AS column_name,
                DATA_TYPE AS data_type,
                (IS_NULLABLE = 'YES') AS is_nullable,
                ORDINAL_POSITION AS ordinal_position
            FROM INFORMATION_SCHEMA."COLUMNS"
            WHERE {self._scope_condition()}
            ORDER BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION
        """)
        if frame.empty:
            return pd.DataFrame(columns=_COLUMN_SCHEMA)
        return self._filter_by_schema(frame)

    def get_views(self) -> pd.DataFrame:
        frame = self.execute(f"""
            SELECT
                TABLE_SCHEMA AS table_schema,
                TABLE_NAME AS table_name,
                VIEW_DEFINITION AS view_definition
            FROM INFORMATION_SCHEMA."VIEWS"
            WHERE {self._scope_condition()}
            ORDER BY TABLE_SCHEMA, TABLE_NAME
        """)
        if frame.empty:
            return pd.DataFrame(columns=_VIEW_SCHEMA)
        return self._filter_by_schema(frame)

    def get_queries(self, hours: int = 24) -> pd.DataFrame:
        """Recent completed queries from ``sys.jobs_recent``.

        Reading it needs a role Dremio does not give every user, so a refusal is
        not an ingestion error: it just means no history is available.

        ``sys.jobs_recent`` is cluster-wide, and on a busy Dremio most of it is
        other people's work (tens of thousands of statements a day). With an
        allowlist, only queries that mention an allowlisted root are returned,
        so the parse step is not spent on unrelated or maintenance statements.
        A root containing ``_`` or ``%`` can over-match here; that only adds
        queries, it never drops one.
        """
        scope = ""
        if self._schema_roots:
            mentions = " OR ".join(
                f"LOWER(query) LIKE '%{root.replace(chr(39), chr(39) * 2)}%'"
                for root in sorted(self._schema_roots)
            )
            scope = f"AND ({mentions})"
        try:
            frame = self.execute(f"""
                SELECT final_state_ts AS end_time, query AS query_text
                FROM sys.jobs_recent
                WHERE status = 'COMPLETED'
                  AND final_state_ts >= TIMESTAMPADD(HOUR, -{int(hours)}, CURRENT_TIMESTAMP)
                  AND LOWER(query) NOT LIKE 'alter %'
                  {scope}
                ORDER BY final_state_ts DESC
            """)
        except Exception:
            logger.debug("Dremio sys.jobs_recent is not readable", exc_info=True)
            return pd.DataFrame(columns=_QUERY_SCHEMA)
        if frame.empty:
            return pd.DataFrame(columns=_QUERY_SCHEMA)
        return frame

    def get_pks(self) -> pd.DataFrame:
        """Dremio declares no primary keys, so this is always empty."""
        return pd.DataFrame(columns=_PK_SCHEMA)

    def get_fks(self) -> pd.DataFrame:
        """Dremio declares no foreign keys, so this is always empty."""
        return pd.DataFrame(columns=_FK_SCHEMA)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def ping(self) -> None:
        """Verify the endpoint, the token, and that metadata is readable."""
        self.execute('SELECT 1 FROM INFORMATION_SCHEMA."SCHEMATA" LIMIT 1')

    def close(self) -> None:
        with self._lock:
            self._reset_connection()
