"""
Error handling middleware for consistent error responses.

Provides centralized error handling with proper logging
and standardized error response format.
"""

import traceback
from datetime import datetime
from typing import Callable

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


async def error_handler_middleware(request: Request, call_next: Callable) -> Response:
    """
    Global error handler middleware.
    
    Catches all unhandled exceptions and returns
    consistent error responses.
    
    Args:
        request: HTTP request
        call_next: Next middleware or route handler
        
    Returns:
        HTTP response (or error response)
    """
    try:
        response = await call_next(request)
        return response
        
    except ValueError as e:
        # Validation errors
        error_response = {
            "error": "Validation Error",
            "detail": str(e),
            "timestamp": datetime.now().isoformat(),
            "path": request.url.path,
            "method": request.method
        }
        
        print(f"❌ Validation Error: {str(e)}")
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=error_response
        )
        
    except FileNotFoundError as e:
        # Resource not found errors
        error_response = {
            "error": "Resource Not Found",
            "detail": str(e),
            "timestamp": datetime.now().isoformat(),
            "path": request.url.path,
            "method": request.method
        }
        
        print(f"❌ Resource Not Found: {str(e)}")
        
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=error_response
        )
        
    except RuntimeError as e:
        # Runtime errors (service issues)
        error_response = {
            "error": "Service Error",
            "detail": str(e),
            "timestamp": datetime.now().isoformat(),
            "path": request.url.path,
            "method": request.method
        }
        
        print(f"❌ Runtime Error: {str(e)}")
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response
        )
        
    except Exception as e:
        # Unexpected errors
        error_trace = traceback.format_exc()
        
        error_response = {
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred",
            "error_type": type(e).__name__,
            "timestamp": datetime.now().isoformat(),
            "path": request.url.path,
            "method": request.method
        }
        
        # Log detailed error with traceback
        print(f"❌ CRITICAL ERROR: {type(e).__name__}: {str(e)}")
        print(f"Traceback:\n{error_trace}")
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response
        )


class ErrorLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for detailed error logging.
    
    Logs all errors with full context for debugging.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with error logging.
        
        Args:
            request: HTTP request
            call_next: Next middleware or route handler
            
        Returns:
            HTTP response
        """
        try:
            response = await call_next(request)
            
            # Log 4xx and 5xx responses
            if response.status_code >= 400:
                correlation_id = getattr(request.state, 'correlation_id', 'N/A')
                print(f"⚠️  Error Response | {correlation_id} | "
                      f"{request.method} {request.url.path} | "
                      f"Status: {response.status_code}")
            
            return response
            
        except Exception as e:
            correlation_id = getattr(request.state, 'correlation_id', 'N/A')
            print(f"💥 Exception in middleware | {correlation_id} | "
                  f"{request.method} {request.url.path} | "
                  f"Error: {type(e).__name__}: {str(e)}")
            raise
