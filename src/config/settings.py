from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    # Required credentials - these must be in .env file
    google_cloud_project: str = Field(..., env="GOOGLE_CLOUD_PROJECT")
    location: str = Field(..., env="LOCATION")
    model_name: str = Field(..., env="MODEL_NAME")
    embedding_model: str = Field(..., env="EMBEDDING_MODEL")  # For Gemini text-embedding-005
    
    # Optional credentials with defaults if not in .env
    max_file_size_mb: int = Field(default=50, env="MAX_FILE_SIZE_MB")
    gcs_staging_bucket: Optional[str] = Field(default=None, env="GCS_STAGING_BUCKET")
    chunk_size: int = Field(default=512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, env="CHUNK_OVERLAP")
    admin_api_key: Optional[str] = Field(default=None, env="AD MIN_API_KEY")
    
    # Monitoring settings
    enable_token_tracking: bool = Field(default=True, env="ENABLE_TOKEN_TRACKING")
    monitoring_storage_backend: str = Field(default="auto", env="MONITORING_STORAGE_BACKEND")
    monitoring_file_path: str = Field(default="monitoring_data", env="MONITORING_FILE_PATH")
    
    # Vertex AI Vector Search settings - required credentials
    vector_search_index_id: str = Field(..., env="VECTOR_SEARCH_INDEX_ID")
    vector_search_endpoint_id: str = Field(..., env="VECTOR_SEARCH_ENDPOINT_ID")
    vector_search_deployed_index_id: str = Field(..., env="VECTOR_SEARCH_DEPLOYED_INDEX_ID")
    
    # Task Queue settings
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    celery_broker_url: str = Field(default="redis://localhost:6379/0", env="CELERY_BROKER_URL")
    celery_result_backend: str = Field(default="redis://localhost:6379/0", env="CELERY_RESULT_BACKEND")
    task_result_expires: int = Field(default=3600, env="TASK_RESULT_EXPIRES")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "validate_default": True
    }


settings = AppSettings()