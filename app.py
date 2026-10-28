
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from data_processor import load_file
from query_engine import generate_sql
from sql_executor import execute_sql


app = FastAPI(
    title="Text-to-SQL AI",
    description="Ask questions about uploaded datasets using natural language.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# Simple in-memory session storage for this portfolio project.
sessions = {}

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".parquet"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB


class QueryRequest(BaseModel):
    upload_id: str
    question: str


@app.get("/")
def read_root():
    return FileResponse("static/index.html")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Save an uploaded file and load it into DuckDB."""

    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        await file.close()
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload CSV, XLSX, XLS, or Parquet.",
        )

    temp_path = None
    con = None

    try:
        # Save the uploaded file to a temporary location.
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp_file:
            temp_path = temp_file.name
            shutil.copyfileobj(file.file, temp_file)

        # Check the saved file size.
        if os.path.getsize(temp_path) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="File is too large. Maximum size is 20 MB.",
            )

        # Load the file using your existing data processor.
        con, schema, preview = load_file(temp_path)

        # Create an ID to identify this uploaded dataset.
        upload_id = str(uuid.uuid4())

        # Keep the connection open for future questions.
        sessions[upload_id] = {
            "con": con,
            "schema": schema,
            "table_name": "uploaded_data",
            "temp_path": temp_path,
            "pending_question": None,
        }

        return {
            "message": "File uploaded and loaded successfully",
            "upload_id": upload_id,
            "filename": filename,
            "columns": schema.to_dict(orient="records"),
            "preview": preview.to_dict(orient="records"),
        }

    except HTTPException:
        if con is not None:
            con.close()
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        raise

    except Exception as exc:
        if con is not None:
            con.close()
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

        raise HTTPException(
            status_code=400,
            detail=f"Could not load file: {exc}",
        ) from exc

    finally:
        await file.close()


@app.post("/query")
def ask_question(request: QueryRequest):
    """Generate SQL, handle clarification, and return query results."""

    session = sessions.get(request.upload_id)

    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Upload session not found. Please upload your file again.",
        )

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    # If clarification is pending, treat the next input as its answer.
    pending = session["pending_question"]

    if pending:
        if question.lower() in {
            "cancel",
            "cancel clarification",
            "new question",
        }:
            session["pending_question"] = None
            return {
                "action": "cancelled",
                "message": "Clarification cancelled. Ask a new question.",
            }

        question_for_model = (
            f"Original question: {pending}\n"
            f"Clarification answer: {question}"
        )
        session["pending_question"] = None
    else:
        question_for_model = question

    try:
        # Step 1: Generate SQL or request clarification.
        plan = generate_sql(
            question=question_for_model,
            schema=session["schema"],
            table_name=session["table_name"],
        )

        if plan["action"] == "clarification":
            # Save the original question for the next request.
            session["pending_question"] = question_for_model

            return {
                "action": "clarification",
                "clarification": plan["clarification"],
                "message": "Please submit your clarification as the next question.",
            }

        # Step 2: Validate and execute the SQL.
        result = execute_sql(
            con=session["con"],
            sql=plan["sql"],
            table_name=session["table_name"],
        )

        # Step 3: Round numeric results to two decimal places.
        result = result.round(2)

        return {
            "action": "result",
            "sql": plan["sql"],
            "columns": result.columns.tolist(),
            "rows": result.to_dict(orient="records"),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Query failed: {exc}",
        ) from exc


@app.delete("/upload/{upload_id}")
def delete_upload(upload_id: str):
    """Close the database and remove a temporary uploaded file."""

    session = sessions.pop(upload_id, None)

    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Upload session not found.",
        )

    session["con"].close()

    temp_path = session.get("temp_path")

    if temp_path and os.path.exists(temp_path):
        os.remove(temp_path)

    return {"message": "Upload session deleted successfully"}
