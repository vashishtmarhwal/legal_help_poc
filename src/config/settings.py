from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    google_cloud_project: str = Field(default="alpine-alpha-469517-g8", env="GOOGLE_CLOUD_PROJECT")
    location: str = Field(default="us-central1", env="LOCATION")
    model_name: str = Field(default="gemini-2.0-flash-lite", env="MODEL_NAME")
    embedding_model: str = Field(default="text-embedding-005", env="EMBEDDING_MODEL")
    max_file_size_mb: int = Field(default=50, env="MAX_FILE_SIZE_MB")
    gcs_staging_bucket: Optional[str] = Field(default=None, env="GCS_STAGING_BUCKET")
    chunk_size: int = Field(default=512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, env="CHUNK_OVERLAP")
    admin_api_key: Optional[str] = Field(default=None, env="ADMIN_API_KEY")
    
    # Monitoring settings
    enable_token_tracking: bool = Field(default=True, env="ENABLE_TOKEN_TRACKING")
    monitoring_storage_backend: str = Field(default="auto", env="MONITORING_STORAGE_BACKEND")
    monitoring_file_path: str = Field(default="monitoring_data", env="MONITORING_FILE_PATH")
    
    # Vertex AI Vector Search settings
    vector_search_index_id: Optional[str] = Field(default="8151183273029009408", env="VECTOR_SEARCH_INDEX_ID")
    vector_search_endpoint_id: Optional[str] = Field(default="2684165169122115584", env="VECTOR_SEARCH_ENDPOINT_ID")
    vector_search_deployed_index_id: str = Field(default="deployed_legal_index_1755658402647", env="VECTOR_SEARCH_DEPLOYED_INDEX_ID")

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = AppSettings()