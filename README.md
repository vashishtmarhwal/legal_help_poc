# Legal Document Assistant API

A FastAPI-powered service for processing legal invoices, extracting structured information, and providing intelligent Q&A capabilities using Google Vertex AI.

## Features

- **Document Processing**: Upload and parse PDF legal invoices
- **Information Extraction**: Extract structured data from invoices
- **Vector Search**: Semantic search using Vertex AI Vector Search
- **Q&A System**: Ask questions about uploaded documents
- **Health Monitoring**: Service health and performance monitoring

## Technology Stack

- **Framework**: FastAPI
- **AI/ML**: Google Vertex AI, Vertex AI Vector Search, LlamaIndex
- **Database**: Google Cloud SQL
- **Document Processing**: PyMuPDF for PDF parsing
- **Storage**: Google Cloud Storage
- **Deployment**: Docker + Google Cloud Run

## Quick Start

The service is deployed on Google Cloud Run:
- **URL**: []
- **Version**: 3.3.0
- **Region**: us-central1