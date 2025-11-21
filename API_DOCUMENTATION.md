# FastAPI Implementation for Atelier2

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install fastapi uvicorn[standard] pydantic httpx python-multipart
# OR use make command
make api-install
```

### 2. Start the API Server
```bash
# Development mode (auto-reload)
make api-dev

# Production mode (4 workers)
make api-prod

# Or run directly
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Access the API
- **Interactive Docs (Swagger UI)**: http://localhost:8000/docs
- **Alternative Docs (ReDoc)**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## 📁 Project Structure

```
Atelier2/
├── api/                          # FastAPI application
│   ├── __init__.py              # Package initialization
│   ├── main.py                  # Main application & configuration
│   ├── routers/                 # API route handlers
│   │   ├── __init__.py
│   │   ├── prediction.py        # Prediction endpoints
│   │   ├── health.py            # Health check endpoints
│   │   └── model.py             # Model info endpoints
│   ├── schemas/                 # Pydantic data models
│   │   ├── __init__.py
│   │   ├── customer.py          # Customer input schemas
│   │   ├── prediction.py        # Prediction response schemas
│   │   ├── health.py            # Health check schemas
│   │   └── model.py             # Model info schemas
│   ├── services/                # Business logic layer
│   │   ├── __init__.py
│   │   └── model_service.py     # ML model service
│   └── middleware/              # Cross-cutting concerns
│       ├── __init__.py
│       ├── logging.py           # Request/response logging
│       ├── timing.py            # Performance monitoring
│       └── error_handler.py     # Error handling
├── tests/                       # Test suite
│   └── test_api.py              # API tests
├── Dockerfile                   # Docker image definition
├── docker-compose.yml           # Docker orchestration
├── .dockerignore               # Docker build exclusions
└── API_DOCUMENTATION.md        # This file
```

## 🎯 Architecture Overview

### Clean Architecture Principles

```
┌─────────────────────────────────────────┐
│         API Layer (FastAPI)             │
│  ┌───────────────────────────────────┐  │
│  │      Routers (HTTP Handlers)      │  │
│  └───────────┬───────────────────────┘  │
│              │                           │
│  ┌───────────▼───────────────────────┐  │
│  │     Schemas (Data Validation)     │  │
│  └───────────┬───────────────────────┘  │
│              │                           │
│  ┌───────────▼───────────────────────┐  │
│  │    Services (Business Logic)      │  │
│  └───────────┬───────────────────────┘  │
│              │                           │
│  ┌───────────▼───────────────────────┐  │
│  │  Model Pipeline (ML Operations)   │  │
│  └───────────────────────────────────┘  │
│                                          │
│  ┌───────────────────────────────────┐  │
│  │   Middleware (Cross-cutting)      │  │
│  │  - Logging                        │  │
│  │  - Timing                         │  │
│  │  - Error Handling                 │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

### Benefits of This Architecture

1. **Separation of Concerns**: Each layer has a single responsibility
2. **Testability**: Easy to test individual components
3. **Maintainability**: Changes are localized to specific layers
4. **Scalability**: Can scale different components independently
5. **Reusability**: Services can be used by multiple routers

## 📡 API Endpoints

### Health & Status

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Comprehensive health check |
| `/health/ready` | GET | Readiness probe (Kubernetes) |
| `/health/live` | GET | Liveness probe (Kubernetes) |
| `/health/startup` | GET | Startup completion check |

### Predictions

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Single customer prediction |
| `/predict/batch` | POST | Batch prediction (up to 1000) |
| `/predict/example` | GET | Get example customer data |

### Model Information

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/model/info` | GET | Full model metadata & metrics |
| `/model/version` | GET | Model version information |
| `/model/features` | GET | Required feature list |
| `/model/metrics` | GET | Performance metrics |

## 💡 Usage Examples

### 1. Single Prediction (cURL)

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "State": "CA",
    "Area_code": 415,
    "International_plan": "No",
    "Voice_mail_plan": "Yes",
    "Account_length": 128,
    "Number_vmail_messages": 25,
    "Total_day_minutes": 265.1,
    "Total_day_calls": 110,
    "Total_day_charge": 45.07,
    "Total_eve_minutes": 197.4,
    "Total_eve_calls": 99,
    "Total_eve_charge": 16.78,
    "Total_night_minutes": 244.7,
    "Total_night_calls": 91,
    "Total_night_charge": 11.01,
    "Total_intl_minutes": 10.0,
    "Total_intl_calls": 3,
    "Total_intl_charge": 2.70,
    "Customer_service_calls": 1
  }'
```

**Response:**
```json
{
  "prediction": "No Churn",
  "churn_probability": 0.12,
  "risk_category": "Low Risk",
  "confidence": 0.88,
  "timestamp": "2025-11-21T10:30:45.123456",
  "model_version": "v1.0"
}
```

### 2. Batch Prediction (Python)

```python
import requests

url = "http://localhost:8000/predict/batch"

batch_data = {
    "customers": [
        {
            "State": "CA",
            "Area_code": 415,
            "International_plan": "No",
            "Voice_mail_plan": "Yes",
            "Account_length": 128,
            "Number_vmail_messages": 25,
            "Total_day_minutes": 265.1,
            "Total_day_calls": 110,
            "Total_day_charge": 45.07,
            "Total_eve_minutes": 197.4,
            "Total_eve_calls": 99,
            "Total_eve_charge": 16.78,
            "Total_night_minutes": 244.7,
            "Total_night_calls": 91,
            "Total_night_charge": 11.01,
            "Total_intl_minutes": 10.0,
            "Total_intl_calls": 3,
            "Total_intl_charge": 2.70,
            "Customer_service_calls": 1
        },
        # ... more customers
    ]
}

response = requests.post(url, json=batch_data)
result = response.json()

print(f"Total customers: {result['total_customers']}")
print(f"High risk: {result['high_risk_count']}")
print(f"Processing time: {result['processing_time_ms']}ms")
```

### 3. Get Model Information

```python
import requests

response = requests.get("http://localhost:8000/model/info")
model_info = response.json()

print(f"Model: {model_info['model_type']}")
print(f"Version: {model_info['model_version']}")
print(f"Accuracy: {model_info['metrics']['accuracy']:.2%}")
print(f"Features: {len(model_info['feature_names'])}")
```

## 🐳 Docker Deployment

### Build and Run with Docker

```bash
# Build image
docker build -t churn-prediction-api:latest .

# Run container
docker run -d \
  --name churn-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  churn-prediction-api:latest
```

### Docker Compose (Recommended)

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

### With Monitoring (Prometheus + Grafana)

```bash
docker-compose --profile monitoring up -d
```

Access:
- **API**: http://localhost:8000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

## 🧪 Testing

### Run API Tests

```bash
# Using pytest
pytest tests/test_api.py -v

# Using make
make test-api
```

### Manual Testing with cURL

```bash
# Health check
curl http://localhost:8000/health

# Get example data
curl http://localhost:8000/predict/example

# Make prediction (use example data)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @example_customer.json
```

## 📊 Monitoring & Observability

### Request Logging

Every request is logged with:
- Correlation ID for tracing
- Processing time
- Status code
- Client IP address

Example log:
```
[2025-11-21T10:30:45] REQUEST | abc-123-def | POST /predict | IP: 192.168.1.1
[2025-11-21T10:30:45] RESPONSE | abc-123-def | Status: 200 | Time: 45.23ms | Level: INFO
```

### Custom Headers

All responses include:
- `X-Correlation-ID`: Unique request identifier
- `X-Process-Time`: Processing time in milliseconds
- `X-Server-Time`: Server timestamp

### Health Checks

Kubernetes-compatible health endpoints:
- **Liveness**: Is the service alive?
- **Readiness**: Can it accept traffic?
- **Startup**: Has initialization completed?

## 🔧 Configuration

### Environment Variables

```bash
# Configuration file path
export ML_CONFIG_PATH=config/config.yaml

# Logging level
export LOG_LEVEL=INFO

# API port
export PORT=8000
```

### config.yaml

API-specific configuration in `config/config.yaml`:

```yaml
api:
  host: "0.0.0.0"
  port: 8000
  workers: 1
  reload: true
  
  cors:
    enabled: true
    allow_origins: ["*"]
  
  batch:
    max_customers: 1000
    timeout_seconds: 30
```

## 🚀 Production Deployment

### Gunicorn with Uvicorn Workers

```bash
gunicorn api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --access-logfile - \
  --error-logfile -
```

### Systemd Service

Create `/etc/systemd/system/churn-api.service`:

```ini
[Unit]
Description=Churn Prediction API
After=network.target

[Service]
Type=notify
User=apiuser
WorkingDirectory=/opt/churn-api
ExecStart=/opt/churn-api/venv/bin/gunicorn \
  api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable churn-api
sudo systemctl start churn-api
```

## 📈 Performance Optimization

### Best Practices Implemented

1. **Async/Await**: FastAPI's async capabilities for non-blocking I/O
2. **Connection Pooling**: Reuse connections for external services
3. **Caching**: Model loaded once and cached in memory
4. **Batch Processing**: Efficient batch prediction endpoint
5. **Request Validation**: Pydantic validates before processing
6. **Response Streaming**: For large responses (future enhancement)

### Benchmarks

Typical performance on standard hardware:
- Single prediction: 20-50ms
- Batch (100 customers): 200-500ms
- Health check: <5ms

## 🔒 Security Considerations

### Implemented

- ✅ Non-root Docker user
- ✅ Input validation (Pydantic)
- ✅ Request size limits
- ✅ CORS configuration
- ✅ Error message sanitization

### Recommended (Production)

- 🔐 API key authentication
- 🔐 Rate limiting per client
- 🔐 HTTPS/TLS encryption
- 🔐 Network security groups
- 🔐 Regular security audits

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Uvicorn Documentation](https://www.uvicorn.org/)
- [Docker Documentation](https://docs.docker.com/)

## 🤝 Contributing

When adding new endpoints:

1. Create schemas in `api/schemas/`
2. Implement business logic in `api/services/`
3. Add routes in `api/routers/`
4. Register router in `api/main.py`
5. Add tests in `tests/`
6. Update this documentation

## 📝 License

MIT License - See LICENSE file for details
