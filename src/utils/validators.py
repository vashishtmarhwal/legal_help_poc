import fitz
from fastapi import HTTPException, UploadFile, status

from ..config import settings

async def validate_pdf_file(file: UploadFile) -> bytes:
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type: {file.content_type}. Please upload a PDF file.",
        )

    file_bytes = await file.read()

    file_size_mb = len(file_bytes) / (1024 * 1024)
    if file_size_mb > settings.max_file_size_mb:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File size ({file_size_mb:.2f} MB) exceeds maximum "
                f"({settings.max_file_size_mb} MB)"
            ),
        )

    try:
        test_doc = fitz.open(stream=file_bytes, filetype="pdf")
        test_doc.close()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="File is not a valid PDF",
        )

    return file_bytes
