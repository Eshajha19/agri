# ML Model Deployment Checklist

Use this checklist before deploying the application with ML models to production.

## Pre-Deployment Setup

### Model Files Preparation
- [ ] All required model files are available:
  - [ ] `yield_model.joblib` (XGBoost main model)
  - [ ] `lstm_yield_model.keras` (LSTM time-series model) - optional but recommended
  - [ ] `lstm_scaler.joblib` (LSTM scaler) - optional but improves accuracy
  - [ ] `models/crop_recommendation.pkl` - optional
  - [ ] `models/fertilizer.pkl` - optional

### Model Signing (Production Only)
- [ ] Generate signing key: `openssl rand -hex 32`
- [ ] Store signing key securely (e.g., environment variable, secrets manager)
- [ ] Sign all production models:
  ```bash
  export MODEL_SIGNING_KEY="<your-key>"
  python -c "from ml.security import sign_model; sign_model('yield_model.joblib')"
  python -c "from ml.security import sign_model; sign_model('lstm_yield_model.keras')"
  ```
- [ ] Verify signature files exist (.sig files)

### Environment Configuration

#### For Docker/Kubernetes Deployments
- [ ] Models copied to `/app/models/` in container
- [ ] No additional environment variables needed
- [ ] `MODEL_SIGNING_KEY` set for production

#### For Render/Railway Deployments
- [ ] `ML_MODEL_PATH` set to model location (if not using `/app/models/`)
- [ ] `ML_MODELS_DIR` set (optional, for generic models)
- [ ] `ML_SCALER_PATH` set for LSTM (optional)
- [ ] `MODEL_SIGNING_KEY` set for production

#### For Custom Deployments
- [ ] Set appropriate environment variables:
  ```bash
  export ML_MODEL_PATH="/path/to/yield_model.joblib"
  export ML_SCALER_PATH="/path/to/lstm_scaler.joblib"
  export MODEL_SIGNING_KEY="<production-key>"
  ```

## Pre-Flight Checks

### Local Verification
- [ ] Run dependency checks:
  ```bash
  python -c "from startup_checks import verify_startup_dependencies; verify_startup_dependencies().log_all()"
  ```
  - [ ] All ML models found
  - [ ] Firebase configured (if used)
  - [ ] Twilio configured (if used)
  - [ ] Weather API configured

### Model Loading Tests
- [ ] Test XGBoost model loading:
  ```bash
  python -c "
  from ml.adapters.xgboost_adapter import XGBoostAdapter
  adapter = XGBoostAdapter()
  adapter.load('yield_model.joblib')
  print('✓ XGBoost model loads successfully')
  "
  ```

- [ ] Test LSTM model loading (if using):
  ```bash
  python -c "
  from ml.adapters.lstm_adapter import LSTMAdapter
  adapter = LSTMAdapter()
  adapter.load('lstm_yield_model.keras', 'lstm_scaler.joblib')
  print('✓ LSTM model loads successfully')
  "
  ```

- [ ] Test generic model loading:
  ```bash
  python -c "
  from ml.model_loader import get_crop_recommendation_model
  model = get_crop_recommendation_model()
  print('✓ Crop recommendation model loads successfully')
  "
  ```

### Signature Verification Tests (Production Only)
- [ ] Test model signature verification:
  ```bash
  export MODEL_SIGNING_KEY="<your-key>"
  python -c "
  from ml.security import verify_and_load_joblib
  model = verify_and_load_joblib('yield_model.joblib')
  print('✓ Model signature verified')
  "
  ```

- [ ] Verify all signature files exist:
  ```bash
  ls -la *.sig
  ```

### FastAPI Startup Test
- [ ] Test full application startup:
  ```bash
  timeout 10 python main.py || true
  ```
  - [ ] No errors in ML warmup logs
  - [ ] "ML warmup completed" message appears
  - [ ] No CRITICAL or ERROR messages related to models

## Deployment Execution

### Container/Platform Deployment
- [ ] Dockerfile/deployment config includes:
  - [ ] All model files in `/app/models/`
  - [ ] `MODEL_SIGNING_KEY` in production secrets
  - [ ] Required environment variables set
  - [ ] Python requirements installed (tensorflow, xgboost, joblib, etc.)

- [ ] Verify after deployment:
  ```bash
  # Check model files
  docker exec <container> ls -la /app/models/
  
  # Check logs for ML warmup
  docker logs <container> | grep "ML warmup"
  ```

### Environment Variable Verification
- [ ] Verify environment variables in deployed environment:
  ```bash
  echo $ML_MODEL_PATH
  echo $ML_MODELS_DIR
  echo $MODEL_SIGNING_KEY
  ```

## Post-Deployment Verification

### Startup Logs
- [ ] Check application startup logs for:
  - [ ] ✓ "ML warmup completed (xgboost)" - Success
  - [ ] ✓ "Firestore reachable" - Database OK
  - [ ] ⚠ "Scaler not found" - Non-critical warning
  - [ ] ✗ "Model file not found" - Critical error
  - [ ] ✗ "HMAC-SHA256 verification FAILED" - Security issue

### API Health Checks
- [ ] Test model inference endpoint:
  ```bash
  curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{
      "Crop": "Rice",
      "CropCoveredArea": 5.0,
      "CHeight": 100,
      "CNext": "Wheat",
      "CLast": "None",
      "CTransp": "Flood",
      "IrriType": "Well",
      "IrriSource": "Well",
      "IrriCount": 5,
      "WaterCov": 75,
      "Season": "Winter"
    }'
  ```
  - [ ] Returns 200 status code
  - [ ] Returns valid JSON with `predicted_ExpYield`

### Logging Verification
- [ ] Application logs show normal operation
- [ ] No repeated model loading errors
- [ ] Model predictions are working

### Database Integration (if used)
- [ ] Verify ML results are being stored
- [ ] Check prediction accuracy against historical data
- [ ] Monitor prediction latency

## Rollback Plan

If models fail after deployment:

1. **Immediate Action**
   - [ ] Check application logs for error messages
   - [ ] Verify model files are accessible
   - [ ] Verify environment variables are set correctly

2. **Quick Fixes**
   - [ ] Restart application service
   - [ ] Verify signatures if using MODEL_SIGNING_KEY
   - [ ] Check model file integrity (corruption check)

3. **Rollback to Previous Version**
   - [ ] Redeploy previous working model version
   - [ ] Set `ML_MODEL_PATH` to previous model location
   - [ ] Verify application comes up healthy

4. **Escalation**
   - [ ] Check if models need retraining
   - [ ] Verify model dependencies (TensorFlow, XGBoost versions)
   - [ ] Contact ML engineering team

## Documentation References

- **ML Deployment Guide**: [ML_DEPLOYMENT_GUIDE.md](ML_DEPLOYMENT_GUIDE.md)
- **Architecture & Issues**: [AGENTS.md](AGENTS.md)
- **Contributing Guidelines**: [CONTRIBUTING.md](CONTRIBUTING.md)
- **Production Readiness**: [PRODUCTION_READINESS_AUDIT.md](PRODUCTION_READINESS_AUDIT.md)

## Sign-Off

- [ ] Pre-deployment verification completed by: _________________ Date: _______
- [ ] Post-deployment verification completed by: _________________ Date: _______
- [ ] Any issues identified: _________________________________

## Quick Reference

### Check Model Status
```bash
# See which models are loaded
python -c "
from ml.registry import ModelRegistry
print('Registered models:', ModelRegistry().list_models())
"
```

### View All Model Paths
```bash
python -c "
from lstm_yield_model import MODEL_PATH, SCALER_PATH
from main import _resolve_yield_model_path
print('Yield Model:', _resolve_yield_model_path())
print('LSTM Model:', MODEL_PATH)
print('LSTM Scaler:', SCALER_PATH)
"
```

### Test Full Pipeline
```bash
python -c "
import logging
logging.basicConfig(level=logging.INFO)
from startup_checks import verify_startup_dependencies
report = verify_startup_dependencies()
print('\nAll checks:', 'PASSED' if report.failed == 0 else 'FAILED')
"
```
