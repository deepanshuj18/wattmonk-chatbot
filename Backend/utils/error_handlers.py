from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from typing import Callable, Dict, Any
import traceback

from utils.logging_utils import app_logger


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""
    pass


class ServiceUnavailable(Exception):
    """Exception raised when a required service is unavailable."""
    pass


class ValidationError(Exception):
    """Exception raised when input validation fails."""
    pass


async def rate_limit_exception_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """
    Handle rate limit exceeded exceptions.
    
    Args:
        request: FastAPI request
        exc: Exception instance
        
    Returns:
        JSON response with error details
    """
    app_logger.warning(f"Rate limit exceeded: {request.client.host} - {request.url.path}")
    return JSONResponse(
        status_code=429,
        content={"detail": str(exc) or "Too many requests. Please try again later."}
    )


async def service_unavailable_exception_handler(request: Request, exc: ServiceUnavailable) -> JSONResponse:
    """
    Handle service unavailable exceptions.
    
    Args:
        request: FastAPI request
        exc: Exception instance
        
    Returns:
        JSON response with error details
    """
    app_logger.error(f"Service unavailable: {str(exc)}")
    return JSONResponse(
        status_code=503,
        content={"detail": str(exc) or "Service temporarily unavailable. Please try again later."}
    )


async def validation_exception_handler(request: Request, exc: ValidationError) -> JSONResponse:
    """
    Handle validation error exceptions.
    
    Args:
        request: FastAPI request
        exc: Exception instance
        
    Returns:
        JSON response with error details
    """
    app_logger.warning(f"Validation error: {str(exc)}")
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc) or "Invalid input data."}
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    Handle HTTP exceptions.
    
    Args:
        request: FastAPI request
        exc: Exception instance
        
    Returns:
        JSON response with error details
    """
    app_logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle general exceptions.
    
    Args:
        request: FastAPI request
        exc: Exception instance
        
    Returns:
        JSON response with error details
    """
    # Log the full traceback for debugging
    app_logger.error(f"Unhandled exception: {str(exc)}")
    app_logger.debug(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again later."}
    )


def register_exception_handlers(app) -> None:
    """
    Register all exception handlers with the FastAPI app.
    
    Args:
        app: FastAPI application instance
    """
    app.add_exception_handler(RateLimitExceeded, rate_limit_exception_handler)
    app.add_exception_handler(ServiceUnavailable, service_unavailable_exception_handler)
    app.add_exception_handler(ValidationError, validation_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)