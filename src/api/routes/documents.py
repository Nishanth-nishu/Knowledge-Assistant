"""
Document management endpoints.
"""

import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

router = APIRouter()


# =============================================================================
# Models
# =============================================================================

class DocumentMetadata(BaseModel):
    """Document metadata."""
    title: str | None = None
    author: str | None = None
    source: str | None = None
    tags: list[str] = Field(default_factory=list)
    custom: dict[str, Any] = Field(default_factory=dict)


class DocumentResponse(BaseModel):
    """Document response model."""
    document_id: str
    filename: str
    content_type: str
    size_bytes: int
    chunk_count: int
    metadata: DocumentMetadata
    created_at: str
    status: str  # "processing", "indexed", "failed"


class DocumentList(BaseModel):
    """List of documents."""
    documents: list[DocumentResponse]
    total: int
    page: int
    page_size: int


class UploadResponse(BaseModel):
    """Upload response."""
    document_id: str
    filename: str
    status: str
    message: str


# =============================================================================
# Endpoints
# =============================================================================

@router.post(
    "/documents/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload Document",
    description="""
    Upload a document for indexing.
    
    Supported formats:
    - PDF (.pdf)
    - Word (.docx, .doc)
    - Text (.txt)
    - HTML (.html)
    - Markdown (.md)
    
    The document will be processed asynchronously.
    Use the document status endpoint to check processing status.
    """,
)
async def upload_document(
    file: UploadFile = File(...),
    metadata: DocumentMetadata | None = None,
) -> UploadResponse:
    """
    Upload a document for processing and indexing.
    """
    from src.config import get_settings
    
    settings = get_settings()
    
    # Validate file extension
    if file.filename:
        ext = file.filename.rsplit(".", 1)[-1].lower()
        if ext not in settings.document.allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type '{ext}' not supported. Allowed: {settings.document.allowed_extensions}",
            )
    
    # Validate file size
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.document.max_upload_size_mb:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {settings.document.max_upload_size_mb}MB",
        )
    
    document_id = str(uuid.uuid4())
    
    # TODO: Save file and queue for processing
    
    return UploadResponse(
        document_id=document_id,
        filename=file.filename or "unknown",
        status="processing",
        message="Document uploaded successfully. Processing started.",
    )


@router.get(
    "/documents",
    response_model=DocumentList,
    summary="List Documents",
    description="List all indexed documents with pagination.",
)
async def list_documents(
    page: int = 1,
    page_size: int = 20,
    status: str | None = None,
    tag: str | None = None,
) -> DocumentList:
    """
    List all documents with optional filtering.
    """
    # TODO: Implement document listing from database
    return DocumentList(
        documents=[],
        total=0,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/documents/{document_id}",
    response_model=DocumentResponse,
    summary="Get Document",
    description="Get details about a specific document.",
)
async def get_document(document_id: str) -> DocumentResponse:
    """
    Get document details by ID.
    """
    # TODO: Implement document retrieval
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Document {document_id} not found",
    )


@router.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Document",
    description="Delete a document and its chunks from the index.",
)
async def delete_document(document_id: str) -> None:
    """
    Delete a document and all its chunks.
    """
    # TODO: Implement document deletion
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Document {document_id} not found",
    )


@router.get(
    "/documents/{document_id}/chunks",
    summary="Get Document Chunks",
    description="Get all chunks for a document.",
)
async def get_document_chunks(
    document_id: str,
    page: int = 1,
    page_size: int = 50,
) -> dict[str, Any]:
    """
    Get chunks for a document.
    """
    # TODO: Implement chunk retrieval
    return {
        "document_id": document_id,
        "chunks": [],
        "total": 0,
        "page": page,
        "page_size": page_size,
    }
