

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.routers import prediction_router, health_router, model_router
from api.middleware.logging import LoggingMiddleware
from api.middleware.timing import TimingMiddleware
from api.middleware.error_handler import error_handler_middleware, ErrorLoggingMiddleware
from api.services.model_service import model_service


# ============================================================================
# APPLICATION LIFECYCLE
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events:
    - Startup: Load model artifacts and configuration
    - Shutdown: Cleanup resources and connections
    """
    # Startup
    try:
        # Load model on startup
        model_service.load(
            config_path="config/config.yaml",
            model_version="v1.0"
        )
        print(f"Model loaded: {model_service.model_version}")
        
    except Exception as e:
        print(f"Warning: Failed to load model: {e}")
        print("API starting in degraded mode (predictions unavailable)")
    
    yield
    
    # Shutdown
    print("Shutting down API...")


# ============================================================================
# APPLICATION FACTORY
# ============================================================================

def create_application() -> FastAPI:
    """
    Application factory.
    
    Creates and configures FastAPI application with all
    middleware, routers, and settings.
    
    Returns:
        Configured FastAPI application
    """
    # Create FastAPI app
    app = FastAPI(
        title="Customer Churn Prediction API",
        description="Machine Learning API for predicting customer churn in telecommunications.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
        license_info={
            "name": "Fedi Fast Api",
        }
    )
    
    # ============================================================================
    # MIDDLEWARE CONFIGURATION
    # ============================================================================
    
    # CORS Middleware - Allow cross-origin requests
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, specify exact origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Correlation-ID", "X-Process-Time"]
    )
    
    # Trusted Host Middleware - Prevent host header attacks
    # Uncomment in production with actual hosts
    # app.add_middleware(
    #     TrustedHostMiddleware,
    #     allowed_hosts=["localhost", "api.example.com"]
    # )
    
    # Custom Middleware (order matters - applied in reverse)
    app.add_middleware(ErrorLoggingMiddleware)
    app.add_middleware(TimingMiddleware)
    app.add_middleware(LoggingMiddleware)
    
    # Global error handler
    app.middleware("http")(error_handler_middleware)
    
    # ============================================================================
    # ROUTER REGISTRATION
    # ============================================================================
    
    # Include routers
    app.include_router(health_router)
    app.include_router(prediction_router)
    app.include_router(model_router)
    
    # ============================================================================
    # ROOT ENDPOINTS
    # ============================================================================
    
    @app.get(
        "/",
        tags=["General"],
        summary="API Root",
        description="Returns API information and available endpoints"
    )
    async def root():
        """Root endpoint with API overview."""
        return {
            "name": "Customer Churn Prediction API",
            "version": "1.0.0",
            "status": "operational",
            "documentation": {
                "swagger_ui": "/docs",
                "redoc": "/redoc",
                "openapi_spec": "/openapi.json"
            },
            "endpoints": {
                "health_check": "/health",
                "predict_single": "/predict",
                "predict_batch": "/predict/batch",
                "example_data": "/predict/example",
                "model_info": "/model/info",
                "model_version": "/model/version",
                "model_features": "/model/features",
                "model_metrics": "/model/metrics"
            },
            "links": {
                "github": "https://github.com/your-org/churn-prediction",
                "documentation": "https://docs.example.com"
            }
        }
    
    @app.get(
        "/docs-redirect",
        include_in_schema=False
    )
    async def docs_redirect():
        """Redirect to interactive documentation."""
        return RedirectResponse(url="/docs")
    
    # ============================================================================
    # EXCEPTION HANDLERS
    # ============================================================================
    
    @app.exception_handler(404)
    async def not_found_handler(request, exc):
        """Custom 404 handler."""
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "error": "Not Found",
                "detail": f"The endpoint {request.url.path} does not exist",
                "hint": "Visit /docs for API documentation"
            }
        )
    
    @app.exception_handler(405)
    async def method_not_allowed_handler(request, exc):
        """Custom 405 handler."""
        return JSONResponse(
            status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
            content={
                "error": "Method Not Allowed",
                "detail": f"The method {request.method} is not allowed for {request.url.path}",
                "hint": "Check API documentation for allowed methods"
            }
        )
    
    return app


# ============================================================================
# APPLICATION INSTANCE
# ============================================================================

# Create application instance
app = create_application()


# ============================================================================
# MAIN (for direct execution)
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info",
        access_log=True
    )
