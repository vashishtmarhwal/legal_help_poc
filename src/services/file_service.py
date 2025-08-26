"""
File handling utilities for extraction workflows
"""

import hashlib
import logging
from typing import Tuple

from fastapi import UploadFile

from ..utils.validators import validate_pdf_file

logger = logging.getLogger(__name__)

class FileService:
    """Service for file handling operations"""

    @staticmethod
    def generate_file_hash(file_bytes: bytes) -> str:
        """
        Generate SHA256 hash of file content for deduplication

        Args:
            file_bytes: Raw file content

        Returns:
            SHA256 hash as hex string
        """
        return hashlib.sha256(file_bytes).hexdigest()

    @staticmethod
    async def process_upload_file(file: UploadFile) -> Tuple[bytes, str, int]:
        """
        Process uploaded file and return content, hash, and size

        Args:
            file: FastAPI UploadFile object

        Returns:
            Tuple of (file_bytes, file_hash, file_size)

        Raises:
            HTTPException if file validation fails
        """
        try:
            file_bytes = await validate_pdf_file(file)

            # Generate hash
            file_hash = FileService.generate_file_hash(file_bytes)
            file_size = len(file_bytes)

            logger.info(f"Processed file {file.filename}: hash={file_hash[:8]}..., size={file_size}")

            return file_bytes, file_hash, file_size

        except Exception as e:
            logger.error(f"Failed to process upload file {file.filename}: {e}")
            raise

# Global service instance
file_service = FileService()
