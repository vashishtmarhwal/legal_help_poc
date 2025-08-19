import logging
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from ...models.responses import BulkParseResponse
from ...services.pdf_service import parse_pdf_with_blocks
from ...utils.validators import validate_pdf_file

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/upload-and-parse/", response_model=BulkParseResponse)
async def upload_and_parse_pdfs(files: List[UploadFile] = File(...)):
    logger.info(f"Processing {len(files)} PDF files for parsing")

    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided",
        )

    results = []
    errors = []
    successful_files = 0

    for file in files:
        try:
            file_bytes = await validate_pdf_file(file)
            parsed_doc = await parse_pdf_with_blocks(file_bytes, file.filename)
            results.append(parsed_doc)
            successful_files += 1
            logger.info(f"Successfully parsed {file.filename}")

        except HTTPException as e:
            error_detail = {
                "filename": file.filename,
                "error": e.detail,
            }
            errors.append(error_detail)
            logger.error(f"Failed to parse {file.filename}: {e.detail}")
        except Exception as e:
            error_detail = {
                "filename": file.filename,
                "error": f"Unexpected error: {e!s}",
            }
            errors.append(error_detail)
            logger.error(f"Unexpected error parsing {file.filename}: {e!s}")

    return BulkParseResponse(
        total_files=len(files),
        successful_files=successful_files,
        failed_files=len(files) - successful_files,
        results=results,
        errors=errors,
    )