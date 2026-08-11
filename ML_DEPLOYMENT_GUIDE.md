# ML Model Deployment Configuration Guide

This guide explains how the ML models are configured for production deployment and how to properly set up your environment.

## Overview

The ML pipeline has been refactored to support flexible model path resolution, enabling models to work in multiple deployment scenarios:

- **Development**: Models in the repository root or current directory
- **Production (Render/Railway)**: Models in `/app/models/` directory
- **Staged/Preview**: Models configured via environment variables
- **Custom**: Any location via `ML_MODEL_PATH` or `ML_MODELS_DIR` environment variables

## Environment Variables

### Primary Model Path Variables

#### `ML_MODEL_PATH` (Highest Priority)
Explicit path to the main XGBoost yield prediction model.

```bash
export ML_MODEL_PATH="/app/models/yield_model.joblib"
```

**Used by:**
- `main.py` - FastAPI server initialization
- `startup_checks.py` - Dependency verification

#### `ML_MODELS_DIR`
Directory containing all ML models. Alternative to individual path variables.

```bash
export ML_MODELS_DIR="/app/models"
```

**Used by:**
- `ml/model_loader.py` - Generic model loading functions
- Crop recommendation and fertilizer models

### Secondary Model Path Variables

#### `ML_MODEL_PATH` (LSTM Model)
Path to the LSTM yield prediction model (alternative format).

```bash
export ML_MODEL_PATH="/app/models/lstm_yield_model.keras"
```

#### `ML_SCALER_PATH`
Path to the LSTM scaler object (optional, for LSTM predictions).

```bash
export ML_SCALER_PATH="/app/models/lstm_scaler.joblib"
```

**Used by:**
- `lstm_yield_model.py` - LSTM inference server
- `ml/adapters/lstm_adapter.py` - LSTM model adapter

## Model Path Resolution (Priority Order)

### Main Yield Model (`yield_model.joblib`)

1. **Environment Variable**: `ML_MODEL_PATH=/path/to/model.joblib`
2. **Production Location**: `/app/models/yield_model.joblib`
3. **Repository Root**: `<repo_root>/yield_model.joblib`
4. **Current Directory**: `./yield_model.joblib`

### LSTM Model (`lstm_yield_model.keras`)

1. **Environment Variable**: `ML_MODEL_PATH=/path/to/lstm_yield_model.keras`
2. **Production Location**: `/app/models/lstm_yield_model.keras`
3. **Repository Root**: `<repo_root>/lstm_yield_model.keras`
4. **Current Directory**: `./lstm_yield_model.keras`

### LSTM Scaler (`lstm_scaler.joblib`)

1. **Environment Variable**: `ML_SCALER_PATH=/path/to/lstm_scaler.joblib`
2. **Production Location**: `/app/models/lstm_scaler.joblib`
3. **Repository Root**: `<repo_root>/lstm_scaler.joblib`
4. **Current Directory**: `./lstm_scaler.joblib`

### Generic Models (Crop Recommendation, Fertilizer)

When using `ml.model_loader.get_model_by_name()`:

1. **Environment Variable**: `ML_MODELS_DIR=/path/to/models`
2. **Production Location**: `/app/models/<model_name>`
3. **Repository Location**: `<repo_root>/models/<model_name>`
4. **Fallback Locations**: Any provided as function arguments
5. **Current Directory**: `./<model_name>`

## Deployment Scenarios

### Scenario 1: Docker/Kubernetes Deployment

**Setup:**
1. Build Docker image including model files
2. Copy models to `/app/models/` in the container
3. No environment variables needed (uses defaults)

**Dockerfile example:**
```dockerfile
FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

# Copy model files to standard location
COPY models/ /app/models/

CMD ["python", "main.py"]
```

### Scenario 2: Render/Railway with Model Upload

**Setup:**
1. Upload model files to `/app/models/` directory
2. No environment variables needed

**File structure:**
```
/app/
├── models/
│   ├── yield_model.joblib
│   ├── lstm_yield_model.keras
│   ├── lstm_scaler.joblib
│   └── ...
└── main.py
```

### Scenario 3: Environment Variable Configuration

**Setup:**
1. Upload models to any persistent storage
2. Set `ML_MODEL_PATH` environment variable

**Environment:**
```bash
export ML_MODEL_PATH="/persistent/storage/models/yield_model.joblib"
export ML_MODELS_DIR="/persistent/storage/models"
export ML_SCALER_PATH="/persistent/storage/models/lstm_scaler.joblib"
```

### Scenario 4: Development with Multiple Models

**Setup:**
1. Clone repository with models in repo root
2. No environment variables needed

**File structure:**
```
/workspaces/agri/
├── yield_model.joblib
├── lstm_yield_model.keras
├── lstm_scaler.joblib
├── models/
│   ├── crop_recommendation.pkl
│   └── fertilizer.pkl
└── main.py
```

## Model Signature Verification (Production Security)

For production deployments, models should be signed to prevent tampering:

### Signing a Model

```python
from ml.security import sign_model

# Set your signing key
export MODEL_SIGNING_KEY="your-secret-signing-key"

# Sign the model
sign_model("yield_model.joblib")  # Creates yield_model.joblib.sig
```

### Verifying Models at Runtime

Models are automatically verified on load:

```python
from ml.security import verify_and_load_joblib

model = verify_and_load_joblib("yield_model.joblib")  # Verifies signature before loading
```

**Environment variable for verification:**
```bash
export MODEL_SIGNING_KEY="your-secret-signing-key"
```

## Startup Verification

The `startup_checks.py` module verifies that all required models are accessible:

```bash
python -c "from startup_checks import verify_startup_dependencies; verify_startup_dependencies()"
```

**Output example:**
```
[dep-check] ml_models: found: yield_model.joblib (/app/models), lstm_yield_model.keras (/app/models) (5ms)
[dep-check] Dependency checks: 1/1 passed, 0 failed, 0 skipped
```

## Debugging Model Loading Issues

### Check which model path is being used

```python
from lstm_yield_model import MODEL_PATH, SCALER_PATH
from main import _resolve_yield_model_path

print(f"LSTM Model: {MODEL_PATH}")
print(f"LSTM Scaler: {SCALER_PATH}")
print(f"Yield Model: {_resolve_yield_model_path()}")
```

### Enable debug logging

```bash
export LOG_LEVEL=DEBUG
python main.py
```

### Check environment variables

```bash
echo "ML_MODEL_PATH: $ML_MODEL_PATH"
echo "ML_MODELS_DIR: $ML_MODELS_DIR"
echo "ML_SCALER_PATH: $ML_SCALER_PATH"
```

### Verify model files exist

```bash
# Check production location
ls -la /app/models/

# Check repo root
ls -la yield_model.joblib lstm_yield_model.keras

# Check custom location
ls -la $ML_MODEL_PATH
```

## Common Issues and Solutions

### Issue: "Model file not found at..."

**Cause:** Models not in expected location

**Solution:**
1. Verify file exists: `ls -la /path/to/model.joblib`
2. Set environment variable: `export ML_MODEL_PATH="/correct/path/model.joblib"`
3. Check startup logs for attempted locations

### Issue: "LSTM model not found"

**Cause:** TensorFlow model in wrong location or not built

**Solution:**
1. Verify: `ls -la /app/models/lstm_yield_model.keras`
2. Set variable: `export ML_MODEL_PATH="/path/to/lstm_yield_model.keras"`
3. Check file extension (.keras vs .h5)

### Issue: "Scaler file not found"

**Cause:** Optional but recommended for LSTM model accuracy

**Solution:**
1. Verify: `ls -la /app/models/lstm_scaler.joblib`
2. Set variable: `export ML_SCALER_PATH="/path/to/lstm_scaler.joblib"`
3. Models will work without scaler but predictions may have different scale

### Issue: "Model signature verification FAILED"

**Cause:** Model file was tampered with or corrupted

**Solution:**
1. Re-sign model: `python -c "from ml.security import sign_model; sign_model('yield_model.joblib')"`
2. Set signing key: `export MODEL_SIGNING_KEY="your-key"`
3. Ensure key is consistent across deployments

## Performance Optimization

### Model Caching

Models are loaded once at startup and cached in memory:

```python
# Happens during FastAPI startup (lifespan event)
# No reload needed for each request
# Reduces latency from ~2s to <10ms per prediction
```

### Scaler Caching

Scalers are also cached:

```python
# Loaded once at startup with model
# Used for preprocessing/postprocessing
```

### Recommended Deployment Settings

```bash
# Production settings
export ENV=production
export ML_MODEL_PATH="/app/models/yield_model.joblib"
export ML_SCALER_PATH="/app/models/lstm_scaler.joblib"
export MODEL_SIGNING_KEY="production-signing-key"

# Development settings (optional)
export ENV=development
export LOG_LEVEL=DEBUG
```

## Monitoring and Logging

### Key Log Messages

```
✓ "ML model loaded into memory successfully" - Model loaded
✓ "ML warmup completed (xgboost)" - All models initialized
⚠ "Scaler not found" - Continuing without scaler (non-critical)
✗ "Model file not found" - Critical error, check deployment
✗ "HMAC-SHA256 verification FAILED" - Security alert
```

### Health Check Endpoint

```bash
curl http://localhost:8000/health
```

Should return model readiness status (if endpoint is exposed).

## Related Documentation

- [AGENTS.md](AGENTS.md) - Architecture and known issues
- [main.py](main.py) - FastAPI initialization with ML warmup
- [ml/security.py](ml/security.py) - Model signing and verification
- [startup_checks.py](startup_checks.py) - Dependency verification
