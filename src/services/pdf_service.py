import logging
import fitz
from fastapi import HTTPException, status

from ..models.responses import ParsedDocument, PageContent, TextBlock
from ..utils.helpers import calculate_file_hash

logger = logging.getLogger(__name__)

async def parse_pdf_with_blocks(file_bytes: bytes, filename: str) -> ParsedDocument:
    try:
        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
        document_content = []

        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            blocks = page.get_text("blocks")

            structured_blocks = []
            for block in blocks:
                if len(block) >= 7 and block[4].strip():
                    structured_blocks.append(
                        TextBlock(
                            x0=block[0],
                            y0=block[1],
                            x1=block[2],
                            y1=block[3],
                            text=block[4].strip(),
                            block_type=block[5],
                            block_no=block[6],
                        ),
                    )

            document_content.append(
                PageContent(page_number=page_num, blocks=structured_blocks),
            )

        total_pages = len(pdf_document)
        pdf_document.close()

        file_hash = calculate_file_hash(file_bytes)

        return ParsedDocument(
            filename=filename,
            total_pages=total_pages,
            content=document_content,
            file_hash=file_hash,
        )

    except Exception as e:
        logger.error(f"Error parsing PDF {filename}: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error parsing PDF: {e!s}",
        )

async def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
        full_text = ""

        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            page_text = page.get_text()
            full_text += f"\n--- Page {page_num + 1} ---\n{page_text}"

        pdf_document.close()
        return full_text.strip()

    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to extract text from PDF: {e!s}",
        )
