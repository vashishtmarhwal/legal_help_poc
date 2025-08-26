import hashlib
import json
import logging
from datetime import datetime, date
from typing import Optional

from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool

from ..config.settings import settings
from ..models.responses import ExtractedInvoice

logger = logging.getLogger(__name__)

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects"""
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super().default(obj)
    
class ExtractionDbService:
    def __init__(self):
        self.connection_pool: Optional[SimpleConnectionPool] = None

    def connect(self):
        """Initialize database connection pool"""
        if not settings.database_url:
            logger.info("DATABASE_URL not configured, extraction caching disabled")
            return

        try:
            # Create connection pool
            self.connection_pool = SimpleConnectionPool(
                1, 10,  # min and max connections
                settings.database_url
            )

            # Test connection and create table if needed
            self._ensure_table_exists()
            logger.info("Connected to extraction database and table verified")
        except Exception as e:
            logger.warning(f"Database connection failed, extraction caching disabled: {e}")
            self.connection_pool = None

    def _ensure_table_exists(self):
        """Ensure the extraction_cache table exists"""
        if not self.connection_pool:
            return

        create_table_sql = """
        CREATE TABLE IF NOT EXISTS extraction_cache (
            id SERIAL PRIMARY KEY,
            file_hash VARCHAR(64) UNIQUE NOT NULL,
            filename VARCHAR(255),
            extraction_result JSONB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            file_size_bytes INTEGER
        );

        CREATE INDEX IF NOT EXISTS idx_file_hash ON extraction_cache (file_hash);
        """

        conn = None
        try:
            conn = self.connection_pool.getconn()
            with conn.cursor() as cursor:
                cursor.execute(create_table_sql)
            conn.commit()
            logger.info("Extraction cache table verified/created")
        except Exception as e:
            if conn:
                conn.rollback()
            logger.warning(f"Failed to create table: {e}")
            raise
        finally:
            if conn:
                self.connection_pool.putconn(conn)

    def disconnect(self):
        """Close database connection pool"""
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("Disconnected from extraction database")

    def generate_file_hash(self, file_content: bytes) -> str:
        """Generate SHA-256 hash of file content"""
        return hashlib.sha256(file_content).hexdigest()

    def get_cached_extraction(self, file_hash: str) -> Optional[ExtractedInvoice]:
        """Retrieve cached extraction result by file hash"""
        if not self.connection_pool:
            return None

        conn = None
        try:
            conn = self.connection_pool.getconn()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                query = """
                    SELECT extraction_result
                    FROM extraction_cache
                    WHERE file_hash = %s
                """
                cursor.execute(query, (file_hash,))
                result = cursor.fetchone()

                if result:
                    logger.info(f"Cache hit for file hash: {file_hash}")
                    extraction_data = result["extraction_result"]
                    return ExtractedInvoice.model_validate(extraction_data)

                logger.debug(f"Cache miss for file hash: {file_hash}")
                return None

        except Exception as e:
            logger.error(f"Error retrieving cached extraction: {e}")
            return None
        finally:
            if conn:
                self.connection_pool.putconn(conn)

    def store_extraction(
        self,
        file_hash: str,
        filename: str,
        file_size: int,
        extraction_result: ExtractedInvoice
    ) -> bool:
        """Store extraction result in cache"""
        if not self.connection_pool:
            return False

        conn = None
        try:
            conn = self.connection_pool.getconn()
            with conn.cursor() as cursor:
                query = """
                    INSERT INTO extraction_cache
                    (file_hash, filename, extraction_result, file_size_bytes)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (file_hash) DO UPDATE SET
                        filename = EXCLUDED.filename,
                        extraction_result = EXCLUDED.extraction_result,
                        file_size_bytes = EXCLUDED.file_size_bytes,
                        created_at = CURRENT_TIMESTAMP
                """

                cursor.execute(query, (
                    file_hash,
                    filename,
                    json.dumps(extraction_result.model_dump(), cls=DateTimeEncoder),
                    file_size
                ))
            conn.commit()
            logger.info(f"Stored extraction for file hash: {file_hash}")
            return True

        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Error storing extraction result: {e}")
            return False
        finally:
            if conn:
                self.connection_pool.putconn(conn)

    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        if not self.connection_pool:
            return {"error": "Database not connected"}

        conn = None
        try:
            conn = self.connection_pool.getconn()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                query = """
                    SELECT
                        COUNT(*) as total_entries,
                        COUNT(DISTINCT filename) as unique_files,
                        AVG(file_size_bytes) as avg_file_size,
                        MIN(created_at) as oldest_entry,
                        MAX(created_at) as newest_entry
                    FROM extraction_cache
                """
                cursor.execute(query)
                result = cursor.fetchone()
                return dict(result) if result else {}

        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}
        finally:
            if conn:
                self.connection_pool.putconn(conn)
# Global instance
extraction_db_service = ExtractionDbService()
