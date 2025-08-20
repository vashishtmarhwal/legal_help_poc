"""Legal Document Assistant API
Built with FastAPI, LangChain, and Google Vertex AI
"""

import logging
import os
from contextlib import asynccontextmanager

import google.auth
import uvicorn
import vertexai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core.node_parser import SimpleNodeParser
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    SafetySetting,
)

from .api import api_router
from .config import settings
import src.dependencies as deps

# Google Cloud credentials should be set via environment variables
# GOOGLE_APPLICATION_CREDENTIALS should point to service account key file
# GOOGLE_CLOUD_PROJECT should be set to your project ID

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Starting Legal Document Assistant API...")

    try:
        # Try to load Google Cloud credentials
        try:
            import os
            creds_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
            if creds_path and os.path.exists(creds_path):
                credentials, project_id = google.auth.default()
                logger.info("Google Cloud credentials loaded successfully")
            else:
                logger.warning("Google Cloud credentials file not found - service will run with limited functionality")
                logger.info("To enable full functionality, provide valid service account credentials")
        except Exception as cred_error:
            logger.warning(f"Google Cloud credentials not available: {type(cred_error).__name__}")
            logger.warning("Service will run with limited functionality")
        
        vertexai.init(project=settings.google_cloud_project, location=settings.location)
        logger.info(f"Vertex AI initialized (project={settings.google_cloud_project})")

        safety_settings = [
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_UNSPECIFIED,
                threshold=HarmBlockThreshold.BLOCK_NONE,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=HarmBlockThreshold.BLOCK_NONE,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=HarmBlockThreshold.BLOCK_NONE,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=HarmBlockThreshold.BLOCK_NONE,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=HarmBlockThreshold.BLOCK_NONE,
            ),
        ]

        deps.model = GenerativeModel(
            model_name=settings.model_name,
            safety_settings=safety_settings,
        )
        logger.info(f"Model {settings.model_name} initialized")

        try:
            logger.info("Using Vertex AI Vector Search for embeddings")

            if not settings.gcs_staging_bucket:
                raise ValueError("GCS_STAGING_BUCKET is required for document storage functionality")

            try:
                from google.cloud import storage
                storage_client = storage.Client(project=settings.google_cloud_project)
                bucket = storage_client.bucket(settings.gcs_staging_bucket)

                list(bucket.list_blobs(max_results=1))
                logger.info(f"GCS bucket {settings.gcs_staging_bucket} is accessible")

            except Exception as bucket_error:
                error_msg = f"GCS bucket access failed: {bucket_error}"
                logger.error(f"{error_msg}")
                raise Exception(error_msg)

            deps.node_parser = SimpleNodeParser.from_defaults(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )

            deps.vector_store_initialized = True
            logger.info("Document processing components initialized")

        except Exception as ve:
            logger.warning(f"Document processing initialization failed: {ve!s}")
            deps.vector_store_initialized = False

        deps.is_initialized = True

    except Exception as e:
        logger.error(f"Startup failed: {e!s}")
        deps.is_initialized = False
        deps.vector_store_initialized = False

    yield

    logger.info("Shutting down Legal Document Assistant API...")


app = FastAPI(
    title="Legal Document Assistant API",
    description="API for parsing legal documents and extracting structured information",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )