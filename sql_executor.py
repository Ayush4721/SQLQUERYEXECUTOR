
import re
import duckdb
import pandas as pd
import numpy as np

def execute_sql(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    table_name: str = "uploaded_data",
) -> pd.DataFrame:
    """
    Validate and execute a read-only SQL query using DuckDB.
    Return the result as a Pandas DataFrame.
    """

    # 1. Validate input
    if not sql or not sql.strip():
        raise ValueError("SQL query cannot be empty.")

    # 2. Validate table name
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_name):
        raise ValueError("Invalid table name.")

    # 3. Parse SQL without executing it
    try:
        statements = con.extract_statements(sql)
    except Exception as exc:
        raise ValueError(f"Invalid SQL syntax: {exc}") from exc

    # Allow exactly one statement
    if len(statements) != 1:
        raise ValueError("Only one SQL statement is allowed.")

    statement = statements[0]

    # 4. Allow only SELECT statements
    if statement.type != duckdb.StatementType.SELECT:
        raise ValueError("Only read-only SELECT queries are allowed.")

    # 5. Check that the query references the uploaded table
    # This is a basic safeguard, not a complete SQL security check.
    if not re.search(
        rf"\b{re.escape(table_name)}\b",
        sql,
        flags=re.IGNORECASE,
    ):
        raise ValueError(
            f"Query must reference the '{table_name}' table."
        )

    # 6. Execute the query
    try:
        result = con.execute(sql).fetchdf()
    except Exception as exc:
        raise ValueError(f"SQL execution failed: {exc}") from exc

   
    # 7. Clean results for JSON serialization
    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.astype(object).where(pd.notna(result), None)

# 8. Return the cleaned result
    return result
