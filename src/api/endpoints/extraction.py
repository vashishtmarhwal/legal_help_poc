import logging
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from ...dependencies import get_ai_model
from ...models.responses import BulkEntityResponse
from ...services.ai_service import extract_entities_with_ai
from ...services.pdf_service import extract_text_from_pdf
from ...utils.validators import validate_pdf_file

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/extract-entities/", response_model=BulkEntityResponse)
async def extract_entities_bulk(files: List[UploadFile] = File(...), model=Depends(get_ai_model)):
    logger.info(f"Processing {len(files)} PDF files for entity extraction")

    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided",
        )

    results = []
    errors = []
    successful_extractions = 0

    for file in files:
        try:
            file_bytes = await validate_pdf_file(file)
            text = await extract_text_from_pdf(file_bytes)

            if not text.strip():
                error_detail = {
                    "filename": file.filename,
                    "error": "No text found in PDF",
                }
                errors.append(error_detail)
                continue

            extracted_data = await extract_entities_with_ai(text=text, model=model)
            results.append(extracted_data)
            successful_extractions += 1
            logger.info(f"Successfully extracted entities from {file.filename}")

        except HTTPException as e:
            error_detail = {
                "filename": file.filename,
                "error": e.detail,
            }
            errors.append(error_detail)
            logger.error(f"Failed to extract entities from {file.filename}: {e.detail}")
        except Exception as e:
            error_detail = {
                "filename": file.filename,
                "error": f"Unexpected error: {e!s}",
            }
            errors.append(error_detail)
            logger.error(f"Unexpected error extracting from {file.filename}: {e!s}")

    return BulkEntityResponse(
        total_files=len(files),
        successful_extractions=successful_extractions,
        failed_extractions=len(files) - successful_extractions,
        results=results,
        errors=errors,
    )