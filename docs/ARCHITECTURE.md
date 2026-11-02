# End-to-End System Architecture

**Project:** Micro-Quantized CNN for Onboard Crater Detection — Chandrayaan-Inspired Lunar Research  
**Document:** System Architecture Specification  

---

## 1. High-Level Architectural Diagram

```mermaid
graph TD
    subgraph Data Pipeline
        D1[NASA LRO NAC Remote Repository] --> D2[Acquisition Engine: src.data.acquisition]
        D2 --> D3[Integrity Validation & SHA-256 Checksums: src.data.validation]
        D3 --> D4[Annotation Parser: Center/Axis to XYXY Boxes: src.data.annotations]
        D4 --> D5[Anti-Leakage Grouped Splitter: src.data.splitting]
        D5 --> D6[PyTorch LunarDataset: src.data.dataset]
    end

    subgraph Deep Learning Core
        D6 --> M1[Model Factory: SSDLite320 + MobileNetV3-Large: src.models.factory]
        M1 --> M2[Training Engine: MultiBox Loss + AdamW: src.training.train]
        M2 --> M3[Atomic Checkpoint Serializer: src.models.checkpoints]
        M3 --> M4[Best Checkpoint: crater_detector_baseline_best.pth - 43MB]
    end

    subgraph Edge Quantization & Benchmarking
        M4 --> Q1[TorchScript Tracing Export: src.models.export]
        Q1 --> Q2[FP32 ONNX Model: crater_detector_fp32.onnx - 14.3MB]
        Q2 --> Q3[ONNX Runtime Dynamic PTQ Engine: src.models.quantization]
        Q3 --> Q4[INT8 ONNX Model: crater_detector_int8.onnx - 4.1MB]
        Q4 --> B1[Statistical Benchmark Harness: src.benchmarking.benchmark]
        B1 --> B2[Benchmark Reports: JSON / CSV: src.benchmarking.reporting]
    end

    subgraph Storage & Presentation Layer
        B2 --> S1[DatabaseManager: SQLite / PostgreSQL: src.storage.database]
        S1 --> S2[Experiment Registry CRUD: src.storage.experiments]
        S2 --> UI[Streamlit Research App: app.py + pages/]
        Q4 --> UI
        Q2 --> UI
    end
```

---

## 2. Component Subsystems

### 2.1 Data Subsystem (`src/data/`)
- **`acquisition.py`:** Handles network downloads with exponential backoff retries, SHA-256 verification, and fallback file catalogs.
- **`annotations.py`:** Parses elliptical crater representations and converts them to axis-aligned bounding boxes.
- **`dataset.py`:** Custom `torch.utils.data.Dataset` implementing bilinear image resizing to $320 \times 320$, ImageNet tensor normalization, and coordinate scaling.
- **`splitting.py`:** Group-based splitting algorithm grouping image crops by observation ID to enforce zero geographic overlap between train, validation, and test splits.
- **`validation.py`:** Validates image file headers, dimensions, and cryptographic signatures.

### 2.2 Deep Learning & Model Subsystem (`src/models/`)
- **`factory.py`:** Configures SSDLite with depthwise separable prediction heads and ImageNet pre-trained MobileNetV3-Large backbone.
- **`checkpoints.py`:** Atomic checkpoint serialization using temporary file writing and renaming to protect against disk corruption.
- **`export.py`:** Traces the PyTorch model graph into an ONNX representation (`dynamo=False`) with embedded Non-Maximum Suppression.
- **`quantization.py`:** Applies ONNX Runtime dynamic PTQ, quantizing weights to `QInt8` while preserving spatial coordinate operations in FP32.
- **`inference.py`:** Unified high-level inference interface (`CraterDetector`) supporting PIL images, file paths, and NumPy arrays with automatic pre/post-processing.

### 2.3 Training & Evaluation Subsystem (`src/training/`)
- **`train.py`:** Training loop orchestrating forward passes, MultiBox loss computation, gradient clipping, and checkpoint evaluation.
- **`evaluate.py`:** Runs validation and test evaluation routines, computing AP@0.50 and mAP@0.50:0.95.
- **`metrics.py`:** Pure NumPy/PyTorch implementations of IoU, Precision, Recall, and Average Precision.

### 2.4 Benchmarking & Reporting Subsystem (`src/benchmarking/`)
- **`benchmark.py`:** Multi-repetition latency and throughput measurement with untimed warmup cycles and CPU memory monitoring.
- **`reporting.py`:** Formats and saves structured benchmark comparison reports in JSON and CSV.

### 2.5 Storage Subsystem (`src/storage/`)
- **`database.py`:** Database abstraction providing default zero-dependency local SQLite storage, with automatic fallback handling if PostgreSQL credentials fail.
- **`experiments.py`:** CRUD operations for logging experiments, metadata, and JSON benchmark payloads.

### 2.6 Presentation Subsystem (`pages/` & `app.py`)
- Built using **Streamlit 1.66.0** with **`st.navigation`**, custom high-contrast dark lunar CSS, interactive Plotly visualizations, and 8 dedicated research pages.
