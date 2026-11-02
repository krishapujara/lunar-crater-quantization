# API & Python Service Architecture Reference

**Architecture Type:** Direct Python Service Architecture with Streamlit Presentation Layer  
**REST API Status:** The core application uses Streamlit and direct Python function calls; it does not expose a custom REST API.  

---

## 1. Architectural Scope

This research system is designed as an **embedded edge ML prototype** for onboard spacecraft computing where intermediate microservice HTTP network stacks (e.g., FastAPI, Flask, REST) introduce unnecessary latency, memory overhead, and security attack surfaces. All pipeline operations are exposed as direct, strongly typed Python service modules.

If remote or programmatic integration is required, developers should import the Python service modules documented below.

---

## 2. Core Python Service Interfaces

### 2.1 Model Inference Service: `src.models.inference.CraterDetector`

Loads PyTorch checkpoints or ONNX models (FP32/INT8) and performs end-to-end crater detection.

```python
from pathlib import Path
from src.models.inference import CraterDetector

detector = CraterDetector(
    model_path=Path("artifacts/quantized/crater_detector_int8.onnx"),
    device="cpu",
    conf_threshold=0.25,
    iou_threshold=0.45
)
```

#### Methods:
- **`predict(image: Union[Path, str, Image.Image, np.ndarray]) -> DetectionResult`**
  - **Inputs:** File path, PIL image, or NumPy RGB array.
  - **Returns:** Strongly typed `DetectionResult` containing:
    - `boxes`: `List[List[float]]` in `[xmin, ymin, xmax, ymax]` pixel coordinates.
    - `scores`: `List[float]` confidence probabilities $[0.0, 1.0]$.
    - `labels`: `List[int]` class identifiers ($1 = \text{crater}$).
    - `inference_time_ms`: Pure forward pass duration.
    - `total_time_ms`: End-to-end execution duration.
    - `model_type`: Model format (`FP32_ONNX`, `INT8_ONNX`, `PYTORCH`).

---

### 2.2 Quantization Service: `src.models.quantization`

Exports PyTorch checkpoints to ONNX and executes dynamic post-training quantization.

```python
from pathlib import Path
from src.models.quantization import quantize_onnx_model

result = quantize_onnx_model(
    input_onnx_path=Path("artifacts/quantized/crater_detector_fp32.onnx"),
    output_onnx_path=Path("artifacts/quantized/crater_detector_int8.onnx"),
    weight_type="QInt8"
)
```

#### Returns: `QuantizationReport`
- `fp32_model_size_mb`: Baseline size in MB.
- `int8_model_size_mb`: Quantized size in MB.
- `percentage_size_reduction`: Compression percentage.
- `quantized_operators`: Dictionary mapping operator names to counts.

---

### 2.3 Experiment Persistence Service: `src.storage.experiments.ExperimentRegistry`

Records scientific training and benchmark runs to local SQLite or PostgreSQL.

```python
from src.storage.database import DatabaseManager
from src.storage.experiments import ExperimentRegistry

db = DatabaseManager(sqlite_path="artifacts/experiments.db")
registry = ExperimentRegistry(db)

# Log experiment
registry.log_experiment(
    experiment_id="EXP-20261017-001",
    model_name="SSDLite320-MobileNetV3",
    quantization_method="PTQ-Dynamic-QInt8",
    metrics={"ap50": 0.0004, "latency_ms": 236.57},
    benchmark_data={...}
)

# Retrieve experiments
all_experiments = registry.list_experiments()
```

---

## 3. Command-Line (CLI) Service Interfaces

All services can be invoked from the command line:

```powershell
# 1. Dataset Acquisition & Validation
.venv\Scripts\python.exe scripts/download_data.py --config configs/default_config.yaml

# 2. Model Baseline Training
.venv\Scripts\python.exe scripts/train_model.py --config configs/default_config.yaml --epochs 8

# 3. ONNX Export & INT8 Quantization
.venv\Scripts\python.exe scripts/quantize_model.py --config configs/default_config.yaml

# 4. Latency & Size Benchmarking
.venv\Scripts\python.exe scripts/benchmark_models.py --config configs/default_config.yaml --repetitions 30
```
