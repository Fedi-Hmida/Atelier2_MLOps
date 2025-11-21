"""
Timing middleware for request performance monitoring.

Tracks and reports request processing times with
detailed breakdown of operation timing.
"""

import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp


class TimingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for timing HTTP requests.
    
    Adds timing headers to responses:
    - X-Process-Time: Total request processing time
    - X-Server-Time: Current server timestamp
    """
    
    def __init__(self, app: ASGIApp):
        """
        Initialize timing middleware.
        
        Args:
            app: FastAPI application instance
        """
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and add timing information.
        
        Args:
            request: HTTP request
            call_next: Next middleware or route handler
            
        Returns:
            HTTP response with timing headers
        """
        # Record start time with high precision
        start_time = time.perf_counter()
        
        # Process request
        response = await call_next(request)
        
        # Calculate processing time
        process_time = (time.perf_counter() - start_time) * 1000  # milliseconds
        
        # Add timing headers
        response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
        response.headers["X-Server-Time"] = str(int(time.time() * 1000))
        
        # Log slow requests (over 1 second)
        if process_time > 1000:
            print(f"⚠️  SLOW REQUEST: {request.method} {request.url.path} "
                  f"took {process_time:.2f}ms")
        
        return response


class PerformanceMonitor:
    """
    Performance monitoring utility.
    
    Tracks request statistics and performance metrics.
    """
    
    def __init__(self):
        """Initialize performance monitor."""
        self.request_count = 0
        self.total_time = 0.0
        self.min_time = float('inf')
        self.max_time = 0.0
        self.slow_requests = 0
    
    def record_request(self, processing_time: float):
        """
        Record a request's processing time.
        
        Args:
            processing_time: Time in milliseconds
        """
        self.request_count += 1
        self.total_time += processing_time
        self.min_time = min(self.min_time, processing_time)
        self.max_time = max(self.max_time, processing_time)
        
        if processing_time > 1000:  # Over 1 second
            self.slow_requests += 1
    
    def get_stats(self) -> dict:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        avg_time = self.total_time / self.request_count if self.request_count > 0 else 0
        
        return {
            "total_requests": self.request_count,
            "average_time_ms": round(avg_time, 2),
            "min_time_ms": round(self.min_time, 2) if self.min_time != float('inf') else 0,
            "max_time_ms": round(self.max_time, 2),
            "slow_requests": self.slow_requests,
            "slow_request_percentage": round(
                (self.slow_requests / self.request_count * 100) if self.request_count > 0 else 0,
                2
            )
        }
    
    def reset(self):
        """Reset all statistics."""
        self.request_count = 0
        self.total_time = 0.0
        self.min_time = float('inf')
        self.max_time = 0.0
        self.slow_requests = 0


# Global performance monitor instance
performance_monitor = PerformanceMonitor()
