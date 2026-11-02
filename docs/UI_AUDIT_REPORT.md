# Streamlit User Interface & Workflow Audit Report

**Application URL:** `http://localhost:8501/`  
**Framework Version:** Streamlit 1.66.0  
**Design Paradigm:** Lunar Dark Glassmorphic Aesthetic (Deep Space Navy `#0a0e1a`, Cyan Glow `#00e5ff`, Slate `#e0e6ed`)  
**Audit Date:** October 17, 2026  

---

## 1. Sidebar & Navigation Bar Defect Resolution

### The Issue Reported by User:
1. *Navigation bar / sidebar color schema was not visible.*
2. *Sidebar links were not working — clicking links led to blank screens.*

### Technical Root Cause:
Streamlit's legacy multi-page directory auto-discovery conflicted with an in-script radio router. When clicking native sidebar links, individual page scripts in `pages/` executed without invoking their `render_...()` functions. Furthermore, without a `.streamlit/config.toml` file, Streamlit defaulted to light theme styles where navigation link text was dark charcoal (`#31333F`) against a `#0c121e` dark background, making text completely invisible.

### Remediation Applied & Verified:
1. Created `.streamlit/config.toml` enforcing `base = "dark"`, `primaryColor = "#00e5ff"`, and `textColor = "#e0e6ed"`.
2. Refactored `app.py` to modern Streamlit `st.navigation` with logical grouping:
   - **Research & Data:** Overview, Dataset Inspection
   - **Model Inference & Benchmarks:** Live Crater Detection, Benchmark Results
   - **Engineering & Diagnostics:** Training History, Experiment Registry, Architecture Details, System Diagnostics
3. Applied targeted CSS to `[data-testid="stSidebarNav"]`:
   - Non-active links: Bright `#e2e8f0` text, distinct border, and dark blue card background.
   - Hover state: Bright `#00e5ff` cyan text, cyan border, and subtle glow.
   - Active page state: Glowing cyan indicator with 4px left highlight border.
4. Added standalone execution blocks (`if __name__ == "__main__":`) to all 8 page modules.

---

## 2. End-to-End Workflow Verification

### Workflow 1: Application Startup
- **Status:** **PASS**
- **Verification:** App starts via `.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501`.
- **Findings:** Server boots in under 3.5 seconds. Custom CSS, font imports (Inter, JetBrains Mono), and sidebar brand header load with zero layout shifts.

### Workflow 2: Dataset Inspection
- **Status:** **PASS**
- **Verification:** Navigated to `pages/dataset_inspection.py`.
- **Findings:**
  - Displays total tile count (20) and annotation count (23,246).
  - Train/Val/Test filter dropdown populates genuine tile names (`M115143943_mosaic_train_small.png`, etc.).
  - "Overlay Ground-Truth Craters" toggle overlays cyan bounding boxes accurately over crater rims.
  - Interactive Plotly histogram visualizes crater major-axis diameter distributions.
  - Missing manifest triggers informative warning rather than crashing.

### Workflow 3: Live Crater Detection (Inference)
- **Status:** **PASS**
- **Verification:** Navigated to `pages/live_inference.py`.
- **Findings:**
  - "Use Test Sample" loads genuine LRO NAC test tile (`M115143943`).
  - Dual detectors (`CraterDetector` FP32 and INT8) execute in parallel upon user trigger.
  - Bounding boxes and confidence probabilities render side by side.
  - Latency breakdown bar chart visualizes preprocessing, pure inference, and postprocessing timings.
  - Detection summary tables allow JSON and CSV export.

### Workflow 4: FP32 versus INT8 Comparison
- **Status:** **PASS**
- **Verification:** Navigated to `pages/benchmarks.py`.
- **Findings:**
  - Summary metrics display verified numbers: FP32 Size (14.29 MB), INT8 Size (4.10 MB, -71.3% delta).
  - Bar charts and box plots compare FP32 vs INT8 latency distributions.
  - Explanatory callouts honestly detail why dynamic PTQ has higher latency on x86 CPUs.

### Workflow 5: Benchmark Laboratory
- **Status:** **PASS**
- **Verification:** Executed benchmark runner with 30 repetitions.
- **Findings:** Benchmark generates verified JSON and CSV reports in `artifacts/benchmarks/`.

### Workflow 6: Architecture Details
- **Status:** **PASS**
- **Verification:** Navigated to `pages/architecture.py`.
- **Findings:**
  - SSDLite320 + MobileNetV3-Large layer hierarchy table renders correctly.
  - ONNX operator transformation chart shows 98 Conv $\rightarrow$ 98 ConvInteger.
  - Detailed scientific notes explain why NMS coordinate ops remain in FP32.

### Workflow 7: System Health & Diagnostics
- **Status:** **PASS**
- **Verification:** Navigated to `pages/system_diagnostics.py`.
- **Findings:**
  - Host OS (Windows 11), CPU model, logical cores (4), and total RAM (11.8 GB) detected live.
  - Artifact matrix checks 9 critical project files and reports green `READY (PASS)` status.
  - Air-gap compliance and offline execution confirmed.

### Workflow 8: Persistence & Experiment Registry
- **Status:** **PASS**
- **Verification:** Navigated to `pages/experiment_registry.py`.
- **Findings:**
  - Connects to SQLite database at `artifacts/experiments.db`.
  - Lists historical experiment `EXP-20261017-040343-b948cd`.
  - Expanding records displays hyperparameters and raw JSON benchmark payloads.
