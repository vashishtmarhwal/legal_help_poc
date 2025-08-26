import logging
import time

from fastapi import APIRouter, Depends, HTTPException, status

from ...dependencies import get_ai_model, get_vector_store_status
from ...models.requests import QARequest
from ...models.responses import QAResponse, SourceReference
from ...services.ai_service import generate_contextual_answer
from ...services.vector_search_qa_service import (
    search_relevant_documents_vector_search, check_vector_search_readiness
)

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/ask/", response_model=QAResponse)
async def ask_question(
    qa_request: QARequest,
    model=Depends(get_ai_model),
    vector_store_initialized: bool = Depends(get_vector_store_status),
):
    start_time = time.time()

    logger.info(
        f"Processing Q&A request: '{qa_request.question[:100]}"
        f"{'...' if len(qa_request.question) > 100 else ''}'"
    )

    # Check if Vector Search is ready for QA operations
    vector_search_ready = await check_vector_search_readiness()
    if not vector_search_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Vector Search service is not available. Please ensure "
                "the index is deployed and contains documents."
            ),
        )

    try:
        # Direct processing - no cache
        logger.info(
            f"Searching Vector Search index "
            f"(max_sources={qa_request.max_sources}, "
            f"threshold={qa_request.similarity_threshold})"
        )
        (relevant_chunks,
         documents_searched) = await search_relevant_documents_vector_search(
            question=qa_request.question,
            max_sources=qa_request.max_sources,
            similarity_threshold=qa_request.similarity_threshold,
        )

        logger.info(f"Generating answer with {len(relevant_chunks)} relevant chunks")
        answer, confidence_score = await generate_contextual_answer(
            question=qa_request.question,
            relevant_chunks=relevant_chunks,
            include_context=qa_request.include_context,
            model=model,
        )

        # Create sources
        sources = []
        for chunk in relevant_chunks:
            text = chunk.get("text", "").strip()
            if len(text) > 200:
                excerpt = text[:200]
                last_space = excerpt.rfind(" ")
                if last_space > 100:
                    excerpt = excerpt[:last_space] + "..."
                else:
                    excerpt = excerpt + "..."
            else:
                excerpt = text

            source = SourceReference(
                document_id=chunk.get("document_id", ""),
                filename=chunk.get("filename", "Unknown Document"),
                chunk_id=chunk.get("chunk_id", ""),
                chunk_index=chunk.get("chunk_index", 0),
                text_excerpt=excerpt,
                similarity_score=chunk.get("similarity_score", 0.0),
                page_number=chunk.get("page_number"),
            )
            sources.append(source)

        processing_time = time.time() - start_time

        logger.info(
            f"Q&A completed: {len(sources)} sources, "
            f"confidence={confidence_score:.2f}, "
            f"time={processing_time:.2f}s"
        )

        return QAResponse(
            question=qa_request.question,
            answer=answer,
            confidence_score=confidence_score,
            sources=sources,
            total_documents_searched=documents_searched,
            processing_time=processing_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Q&A request failed: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Q&A processing failed: {e!s}",
        )
