
import duckdb
import pandas as pd
import re
from pathlib import Path


def load_file(file_path: str, table_name: str = "uploaded_data"):
    """Load a supported file into an in-memory DuckDB database."""

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")

    # Validate table name before using it in SQL
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_name):
        raise ValueError("Invalid table name.")

    extension = file_path.suffix.lower()

    if extension == ".csv":
        df = pd.read_csv(file_path)

    elif extension in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)

    elif extension == ".parquet":
        df = pd.read_parquet(file_path)

    else:
        raise ValueError(
            "Unsupported file format. Use CSV, XLSX, XLS or Parquet."
        )

    if df.empty:
        raise ValueError("The file is empty or contains no data.")

    # Create an in-memory DuckDB database
    con = duckdb.connect(database=":memory:")

    try:
        con.register("uploaded_df", df)

        con.execute(
            f"CREATE TABLE {table_name} AS "
            "SELECT * FROM uploaded_df"
        )

        con.unregister("uploaded_df")

        schema = con.execute(
            f"DESCRIBE {table_name}"
        ).fetchdf()

        preview = con.execute(
            f"SELECT * FROM {table_name} LIMIT 10"
        ).fetchdf()

        return con, schema, preview

    except Exception:
        con.close()
        raise


if __name__ == "__main__":
    con, schema, preview = load_file("data/sample_data.csv")

    print("\nTable Schema:")
    print(schema)

    print("\nData Preview:")
    print(preview)

    print("\nTesting SQL Query:")
    result = con.execute(
        "SELECT * FROM uploaded_data WHERE age > 30"
    ).fetchdf()

    print(result)

    con.close()
