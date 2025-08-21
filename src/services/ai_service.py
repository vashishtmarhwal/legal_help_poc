import json
import logging
from typing import Dict, List, Optional, Tuple

from fastapi import HTTPException, status
from vertexai.generative_models import GenerationConfig

from ..models.responses import ExtractedInvoice

logger = logging.getLogger(__name__)


async def extract_entities_with_ai(text: str, model) -> ExtractedInvoice:
    if not model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI model is not available",
        )

    prompt = f"""Extract the following information from this legal invoice text and return as a SINGLE JSON object:

Required fields:
- vendor_name: Law firm or vendor name
- invoice_number: Invoice number/ID
- invoice_date: Invoice date
- professional_fees: Subtotal before tax/discounts
- discounts: Total discounts applied
- tax_amount: Total tax charged
- total_amount: Final total due
- line_items: Array of billing entries with:
  - date: Work performed date
  - timekeeper_name: Person who did work
  - timekeeper_role: Title (Partner, Associate, etc.)
  - description: Work description
  - hours_worked: Hours billed
  - total_spent: Amount for line item

Return ONLY a single JSON object (not an array). Use null for missing fields.

Text:
{text[:30000]}"""

    try:
        generation_config = GenerationConfig(
            temperature=0.1,
            top_p=0.95,
            max_output_tokens=8192,
            response_mime_type="application/json",
        )

        response = model.generate_content(
            prompt,
            generation_config=generation_config,
        )

        # Track tokens immediately after response
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            from ..monitoring.simple_token_counter import simple_counter
            total_tokens = response.usage_metadata.total_token_count
            simple_counter.total_tokens += total_tokens
            simple_counter.request_count += 1
            logger.info(f"Tokens tracked - extract_entities: {total_tokens}, Session Total: {simple_counter.total_tokens}")

        json_text = response.text.strip()
        for prefix in ["```json", "```", "json"]:
            json_text = json_text.removeprefix(prefix)
        json_text = json_text.removesuffix("```")
        json_text = json_text.strip()

        parsed_json = json.loads(json_text)

        if isinstance(parsed_json, list):
            if len(parsed_json) > 0:
                parsed_json = parsed_json[0]
            else:
                parsed_json = {}

        extracted_data = ExtractedInvoice.model_validate(parsed_json)

        fields_found = sum([
            extracted_data.vendor_name is not None,
            extracted_data.invoice_number is not None,
            extracted_data.invoice_date is not None,
            extracted_data.total_amount is not None,
            len(extracted_data.line_items) > 0,
        ])
        extracted_data.extraction_confidence = fields_found / 5.0

        return extracted_data

    except Exception as e:
        logger.error(f"Entity extraction failed: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Entity extraction failed: {e!s}",
        )


async def generate_contextual_answer(question: str, relevant_chunks: List[Dict], include_context: bool, model) -> Tuple[str, float]:
    if not model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI model is not available",
        )

    try:
        context_parts = []
        if relevant_chunks:
            context_parts.append("Based on the following relevant document excerpts:")
            context_parts.append("")

            for i, chunk in enumerate(relevant_chunks, 1):
                filename = chunk.get("filename", "Unknown Document")
                text = chunk.get("text", "").strip()
                similarity = chunk.get("similarity_score", 0.0)

                context_parts.append(f"[Source {i}: {filename} (Relevance: {similarity:.2f})]")
                context_parts.append(text)
                context_parts.append("")

        context = "\n".join(context_parts) if context_parts else "No relevant context found."

        if include_context and relevant_chunks:
            prompt = f"""You are a legal document assistant. Answer the user's question based on the provided document context. 

Context from Legal Documents:
{context}

User Question: {question}

Instructions:
1. Provide a comprehensive answer based primarily on the provided context
2. If the context doesn't fully address the question, clearly state what information is missing
3. Cite specific sources when referencing information
4. Use professional legal language appropriate for the context
5. If no relevant information or chunks are found, clearly state this limitation and do not provide a generic answer

Answer:"""
        else:
            prompt = f"""You are a legal document assistant. Answer the following question using your general knowledge about legal matters:

User Question: {question}

Instructions:
1. Provide a helpful and accurate answer
2. Use professional legal language
3. If you're uncertain about legal specifics, recommend consulting with a qualified attorney

Answer:"""

        generation_config = GenerationConfig(
            temperature=0.2,
            top_p=0.95,
            max_output_tokens=2048,
        )

        response = model.generate_content(
            prompt,
            generation_config=generation_config,
        )

        # Track tokens immediately after response
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            from ..monitoring.simple_token_counter import simple_counter
            total_tokens = response.usage_metadata.total_token_count
            simple_counter.total_tokens += total_tokens
            simple_counter.request_count += 1
            logger.info(f"Tokens tracked - generate_answer: {total_tokens}, Session Total: {simple_counter.total_tokens}")

        answer = response.text.strip()

        confidence_score = 0.5

        if relevant_chunks:
            avg_similarity = sum(chunk.get("similarity_score", 0.0) for chunk in relevant_chunks) / len(relevant_chunks)
            confidence_score += avg_similarity * 0.4

            if len(relevant_chunks) > 1:
                confidence_score += 0.1

        confidence_score = min(max(confidence_score, 0.0), 1.0)

        logger.info(f"Generated answer with confidence: {confidence_score:.2f}")
        return answer, confidence_score

    except Exception as e:
        logger.error(f"Answer generation failed: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Answer generation failed: {e!s}",
        )


