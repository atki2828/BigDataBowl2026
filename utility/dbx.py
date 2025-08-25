"""Databricks SQL client for analysis libraries and Streamlit apps.

Design goals
------------
- Zero hard-coded secrets in code. Defaults to environment variables.
- Simple, DB-API–style usage with safe parameterized queries.
- Convenience helpers to return results as Pandas or Polars DataFrames.
- Lightweight retry wrapper for transient network hiccups.
- Context-manager support so connections/cursors are cleaned up.

Environment variables
---------------------
- DATABRICKS_SERVER_HOSTNAME
- DATABRICKS_HTTP_PATH
- DATABRICKS_TOKEN

Optionally, you can pass these explicitly to the constructor. In Streamlit,
prefer setting them via st.secrets and passing them in.
"""

from __future__ import annotations

import contextlib
import os
import time
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)

from databricks import sql as dbsql

Params = Optional[Union[Sequence[Any], Mapping[str, Any]]]
Row = Tuple[Any, ...]


@dataclass(frozen=True)
class DatabricksCredentials:
    """Immutable credentials container.

    Prefer environment variables; explicitly pass in Streamlit using st.secrets.
    """

    server_hostname: str
    http_path: str
    access_token: str

    @staticmethod
    def from_env() -> "DatabricksCredentials":
        return DatabricksCredentials(
            server_hostname=os.environ["DATABRICKS_SERVER_HOSTNAME"],
            http_path=os.environ["DATABRICKS_HTTP_PATH"],
            access_token=os.environ["DATABRICKS_TOKEN"],
        )


class DatabricksSQLClient:
    """Thin wrapper around the Databricks SQL connector with DataFrame helpers.

    Examples
    --------
    Basic usage:
    >>> client = DatabricksSQLClient()  # picks up env vars
    >>> df = client.query_to_pd("SELECT 1 AS x")

    With parameters:
    >>> df = client.query_to_pd(
    ...     "SELECT * FROM workspace.bigdatabowl2024.games WHERE season = %(yr)s LIMIT 10",
    ...     params={"yr": 2024},
    ... )

    Using as a context manager:
    >>> with DatabricksSQLClient() as client:
    ...     pl_df = client.query_to_pl("SELECT * FROM some.table LIMIT 5")

    Streamlit secrets (recommended for deployed apps):
    >>> import streamlit as st
    >>> client = DatabricksSQLClient(
    ...     server_hostname=st.secrets["DATABRICKS_SERVER_HOSTNAME"],
    ...     http_path=st.secrets["DATABRICKS_HTTP_PATH"],
    ...     access_token=st.secrets["DATABRICKS_TOKEN"],
    ... )
    """

    def __init__(
        self,
        server_hostname: Optional[str] = None,
        http_path: Optional[str] = None,
        access_token: Optional[str] = None,
        *,
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
        session_configuration: Optional[Dict[str, Any]] = None,
        connect_timeout: int = 30,
        retries: int = 2,
        retry_backoff_seconds: float = 1.0,
    ) -> None:
        creds = (
            DatabricksCredentials(server_hostname, http_path, access_token)
            if (server_hostname and http_path and access_token)
            else DatabricksCredentials.from_env()
        )

        self._server_hostname = creds.server_hostname
        self._http_path = creds.http_path
        self._access_token = creds.access_token
        self._catalog = catalog
        self._schema = schema
        self._session_configuration = session_configuration or {}
        self._connect_timeout = connect_timeout
        self._retries = retries
        self._retry_backoff_seconds = retry_backoff_seconds
        self._conn = None  # type: Optional[dbsql.Connection]

    # ---- context manager API ----
    def __enter__(self) -> "DatabricksSQLClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover
        self.close()

    # ---- connection lifecycle ----
    def connect(self) -> None:
        """Open a DB connection if one is not already open."""
        if self._conn is not None:
            return
        self._conn = dbsql.connect(
            server_hostname=self._server_hostname,
            http_path=self._http_path,
            access_token=self._access_token,
            catalog=self._catalog,
            schema=self._schema,
            user_agent_entry="dbxutils/DatabricksSQLClient",
            timeout=self._connect_timeout,
            session_configuration=self._session_configuration,
        )

    def close(self) -> None:
        """Close the DB connection if open."""
        if self._conn is not None:
            with contextlib.suppress(Exception):
                self._conn.close()
            self._conn = None

    # ---- core execution helpers ----
    def _ensure_conn(self) -> dbsql.Connection:
        if self._conn is None:
            self.connect()
        assert self._conn is not None
        return self._conn

    def execute(
        self, query: str, params: Params = None, *, max_rows: Optional[int] = None
    ) -> Tuple[List[Row], List[Tuple[str, Any]]]:
        """Execute a SQL statement and fetch all rows.

        Parameters
        ----------
        query : str
            SQL statement. Use DB-API paramstyle "pyformat" or positional placeholders.
            Example: "SELECT * FROM t WHERE season = %(yr)s" with params={"yr": 2024}.
        params : sequence or mapping, optional
            Parameters for the SQL. Passed directly to cursor.execute.
        max_rows : int, optional
            If provided, stops fetching after this many rows.

        Returns
        -------
        rows : list of tuples
        description : list of (name, type_code) for columns (DB-API cursor.description shape)
        """
        last_err: Optional[Exception] = None
        for attempt in range(self._retries + 1):
            try:
                conn = self._ensure_conn()
                with conn.cursor() as cur:
                    (
                        cur.execute(query, params)
                        if params is not None
                        else cur.execute(query)
                    )

                    rows: List[Row] = []
                    if max_rows is None:
                        rows = cur.fetchall()
                    else:
                        remaining = max_rows
                        while remaining > 0:
                            batch = cur.fetchmany(size=min(remaining, 1000))
                            if not batch:
                                break
                            rows.extend(batch)
                            remaining -= len(batch)

                    description = cur.description or []
                    return rows, [(col[0], col[1]) for col in description]
            except Exception as e:  # transient hiccups
                last_err = e
                if attempt < self._retries:
                    time.sleep(self._retry_backoff_seconds * (2**attempt))
                    # best-effort reconnect
                    self.close()
                    continue
                raise
        # Should never reach here
        if last_err:
            raise last_err
        return [], []

    # ---- DataFrame convenience ----
    def query_to_pd(
        self, query: str, params: Params = None, *, max_rows: Optional[int] = None
    ):
        """Return result as a pandas DataFrame.

        This imports pandas lazily to avoid a hard dependency for non-pandas users.
        """
        rows, description = self.execute(query, params=params, max_rows=max_rows)
        try:
            import pandas as pd  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "pandas is required for query_to_pd(). `pip install pandas`. "
            ) from e

        colnames = [c[0] for c in description]
        return pd.DataFrame.from_records(rows, columns=colnames)

    def query_to_pl(
        self, query: str, params: Params = None, *, max_rows: Optional[int] = None
    ):
        """Return result as a Polars DataFrame.

        Attempts the fast path via PyArrow if available, else converts from pandas.
        """
        try:
            import polars as pl  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "polars is required for query_to_pl(). `pip install polars`. "
            ) from e

        # Try Arrow path if connector supports it
        use_arrow = False
        try:
            # Some versions support cursor.fetchallarrow(); we attempt it in a guarded block.
            conn = self._ensure_conn()
            with conn.cursor() as cur:
                if params is not None:
                    cur.execute(query, params)
                else:
                    cur.execute(query)
                if hasattr(cur, "fetchallarrow"):
                    arrow_table = cur.fetchallarrow()
                    use_arrow = True
                    return pl.from_arrow(arrow_table)
        except Exception:
            # Fall back to regular path below
            pass

        # Fallback to rows/columns
        rows, description = self.execute(query, params=params, max_rows=max_rows)
        colnames = [c[0] for c in description]
        return pl.DataFrame(rows, schema=colnames)

    # ---- utilities ----
    def set_default_catalog_schema(
        self, *, catalog: Optional[str] = None, schema: Optional[str] = None
    ) -> None:
        """Update default catalog/schema for future connections.
        This takes effect on the next (re)connect.
        """
        self._catalog = catalog or self._catalog
        self._schema = schema or self._schema
        # force reconnect to apply
        if self._conn is not None:
            self.close()
            self.connect()


if __name__ == "__main__":  # pragma: no cover
    q = """
        SELECT *
        FROM workspace.bigdatabowl2024.games
        LIMIT 4
        """
    client = DatabricksSQLClient()
    print(client.query_to_pl(q).head())
