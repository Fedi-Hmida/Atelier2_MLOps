"""
Logging middleware for API requests and responses.

Provides comprehensive request/response logging with
correlation IDs, timing, and structured logging.
"""

import time
import uuid
from datetime import datetime
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging HTTP requests and responses.
    
    Logs:
    - Request method, path, query params
    - Response status code
    - Request processing time
    - Client IP address
    - Correlation ID for request tracing
    """
    
    def __init__(self, app: ASGIApp):
        """
        Initialize logging middleware.
        
        Args:
            app: FastAPI application instance
        """
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and log details.
        
        Args:
            request: HTTP request
            call_next: Next middleware or route handler
            
        Returns:
            HTTP response
        """
        # Generate correlation ID for request tracing
        correlation_id = str(uuid.uuid4())
        request.state.correlation_id = correlation_id
        
        # Record start time
        start_time = time.time()
        
        # Extract request details
        method = request.method
        path = request.url.path
        query_params = dict(request.query_params)
        client_ip = request.client.host if request.client else "unknown"
        
        # Log request
        print(f"[{datetime.now().isoformat()}] REQUEST | {correlation_id} | "
              f"{method} {path} | IP: {client_ip}")
        
        if query_params:
            print(f"  Query params: {query_params}")
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            process_time = (time.time() - start_time) * 1000  # milliseconds
            
            # Add custom headers
            response.headers["X-Correlation-ID"] = correlation_id
            response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
            
            # Log response
            status_code = response.status_code
            log_level = "INFO" if status_code < 400 else "WARNING" if status_code < 500 else "ERROR"
            
            print(f"[{datetime.now().isoformat()}] RESPONSE | {correlation_id} | "
                  f"Status: {status_code} | Time: {process_time:.2f}ms | Level: {log_level}")
            
            return response
            
        except Exception as e:
            # Calculate processing time even for errors
            process_time = (time.time() - start_time) * 1000
            
            # Log error
            print(f"[{datetime.now().isoformat()}] ERROR | {correlation_id} | "
                  f"{method} {path} | Time: {process_time:.2f}ms | Error: {str(e)}")
            
            # Re-raise exception to be handled by error handler
            raise
