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
from llama_index.core import Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
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

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(os.path.dirname(os.path.dirname(__file__)), "credentials.json")

credentials, project_id = google.auth.default()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Starting Legal Document Assistant API...")

    try:
        vertexai.init(project=settings.google_cloud_project, location=settings.location)
        logger.info(f"✅ Vertex AI initialized (project={settings.google_cloud_project})")

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
        logger.info(f"✅ Model {settings.model_name} initialized")

        try:
            Settings.embed_model = HuggingFaceEmbedding(
                model_name=settings.embedding_model,
                trust_remote_code=True,
            )
            logger.info(f"✅ Embedding model {settings.embedding_model} configured")

            if not settings.gcs_staging_bucket:
                raise ValueError("GCS_STAGING_BUCKET is required for document storage functionality")

            try:
                from google.cloud import storage
                storage_client = storage.Client(project=settings.google_cloud_project)
                bucket = storage_client.bucket(settings.gcs_staging_bucket)

                list(bucket.list_blobs(max_results=1))
                logger.info(f"✅ GCS bucket {settings.gcs_staging_bucket} is accessible")

            except Exception as bucket_error:
                error_msg = f"GCS bucket access failed: {bucket_error}"
                logger.error(f"❌ {error_msg}")
                logger.error("❌ Document storage requires GCS bucket access. Please ensure:")
                logger.error(f"   1. Create the bucket: gsutil mb gs://{settings.gcs_staging_bucket}")
                logger.error("   2. Grant service account permissions:")
                logger.error(f"      gcloud projects add-iam-policy-binding {settings.google_cloud_project} \\")
                logger.error("        --member='serviceAccount:la-datatonic-api-runner@alpine-alpha-469517-g8.iam.gserviceaccount.com' \\")
                logger.error("        --role='roles/storage.admin'")
                logger.error("   3. Or use an existing accessible bucket in .env")
                raise Exception(error_msg)

            deps.node_parser = SimpleNodeParser.from_defaults(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )

            deps.vector_store_initialized = True
            logger.info("✅ Document processing components initialized")

        except Exception as ve:
            logger.warning(f"⚠️ Document processing initialization failed: {ve!s}")
            deps.vector_store_initialized = False

        deps.is_initialized = True

    except Exception as e:
        logger.error(f"❌ Startup failed: {e!s}")
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