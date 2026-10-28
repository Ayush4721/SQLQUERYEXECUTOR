
import os
import re
import json

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal

from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()


class QueryPlan(BaseModel):
    """Structured response from the LLM."""

    action: Literal["sql", "clarification"] = Field(
        description="Generate SQL or ask the user for clarification."
    )

    sql: str = Field(
        description="A single read-only DuckDB SQL query, or empty string."
    )

    clarification: str = Field(
        description="A clarification question, or empty string."
    )


def generate_sql(
    question: str,
    schema: pd.DataFrame,
    table_name: str = "uploaded_data",
) -> dict:
    """
    Generate DuckDB SQL or ask for clarification
    based on the user's natural-language question.
    """

    # 1. Validate user input
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if schema.empty:
        raise ValueError("Table schema cannot be empty.")

    # 2. Validate table name
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_name):
        raise ValueError("Invalid table name.")

    required_columns = {"column_name", "column_type"}

    if not required_columns.issubset(schema.columns):
        raise ValueError(
            "Schema must contain column_name and column_type."
        )

    # 3. Read Gemini configuration
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is missing. Add it to your .env file."
        )

    model_name = os.getenv(
        "GEMINI_MODEL",
        "gemini-2.5-flash",
    )

    # 4. Convert schema to readable information
    schema_info = schema[
        ["column_name", "column_type"]
    ].to_dict(orient="records")

    schema_text = json.dumps(schema_info, indent=2)

    # 5. Initialize Gemini through LangChain
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0,
        google_api_key=api_key,
    )

    # 6. Require a structured response
    structured_llm = llm.with_structured_output(QueryPlan)

    # 7. Give Gemini instructions
    prompt = f"""
You are a Text-to-SQL assistant that generates DuckDB SQL.

TABLE NAME:
{table_name}

TABLE SCHEMA:
{schema_text}

USER QUESTION:
{question}

INSTRUCTIONS:

1. Understand what information the user wants.

2. If the question is clear and answerable using the schema:
   - Set action to "sql".
   - Generate one DuckDB SELECT query.
   - Leave clarification empty.

3. If the question is ambiguous or missing an essential detail:
   - Set action to "clarification".
   - Ask one concise question.
   - Leave sql empty.

4. Example:
   Question: "Which item has the maximum number?"
   If both price and quantity_sold exist, ask which metric
   the user means instead of guessing.

5. Use only the supplied table and columns.
   Never invent column names.

6. Use DuckDB SQL syntax.

7. Generate read-only queries only. Do not generate
   INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, COPY,
   ATTACH, or other statements that modify data or access
   external resources.

8. Do not assume that a column represents sales, revenue,
   ratings, dates, or other meanings unless supported by
   the schema and the user's question.

9. Do not invent missing date ranges or filters.
   Ask for clarification if an essential detail is missing.

10. Do not ask unnecessary questions when the user's intent
    is clear.

11. Return only the fields defined by the response schema.

12. For AVG, division, and other decimal calculations,
    use ROUND(value, 2) when appropriate.
    Give calculated columns meaningful aliases.
    
13. When the user asks for the product/item with the
    highest or lowest value of a metric, return both
    the product/item column and the metric column.
    Give the metric column a meaningful alias when appropriate.
    
    
14. When a previous question and a clarification answer are provided,
    use the clarification answer to resolve the ambiguity in the
    original question. Do not treat the clarification answer as a
    separate question.
    
15. If the clarification answer matches a column name in the supplied
    schema, use that column as the intended metric when appropriate.
    
    
16. Handle numeric values stored as text:
    - Uploaded CSV or Excel files may contain numeric values stored as strings, such as '24,269' or '1,250'.
    - When numeric operations or numeric output are required for such columns, remove commas before converting the values.
    - Use DuckDB-compatible expressions such as TRY_CAST(REPLACE(column_name, ',', '') AS BIGINT) for integers and TRY_CAST(REPLACE(column_name, ',', '') AS DOUBLE) for decimals.
    - Use the actual column names from the supplied schema.
    - TRY_CAST should be used when conversion may fail, so invalid values return NULL instead of causing the query to fail.
    - Do not convert text columns that are not intended to be numeric.
    - Preserve the original column value when numeric conversion is unnecessary.
"""
    # 8. Generate the query plan
    result = structured_llm.invoke(prompt)

    # 9. Validate response consistency
    if result.action == "sql":
        if not result.sql.strip():
            raise ValueError(
                "The model selected SQL but returned no query."
            )

        if result.clarification.strip():
            raise ValueError(
                "Invalid response: SQL action included clarification."
            )

    elif result.action == "clarification":
        if not result.clarification.strip():
            raise ValueError(
                "The model requested clarification without a question."
            )

        if result.sql.strip():
            raise ValueError(
                "Invalid response: clarification action included SQL."
            )

    # 10. Return a simple dictionary for other modules
    return result.model_dump()
