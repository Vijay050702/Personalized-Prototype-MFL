from typing import Any

from fastapi.responses import JSONResponse


def success_response(
    data: Any, message: str = "Success", status_code: int = 200
) -> JSONResponse:
    return JSONResponse(
        content={"status": "success", "message": message, "data": data},
        status_code=status_code,
    )


def error_response(
    message: str, status_code: int = 400, details: dict | None = None
) -> JSONResponse:
    return JSONResponse(
        content={"status": "error", "message": message, "details": details},
        status_code=status_code,
    )
