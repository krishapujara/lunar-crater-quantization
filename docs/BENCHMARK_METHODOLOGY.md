# Benchmark Methodology & Evaluation Integrity

**Project:** Micro-Quantized CNN for Onboard Crater Detection  
**Evaluation Standard:** Aerospace-grade repeatability, transparent hardware accounting, and metric consistency  
**Primary Benchmark Record:** `artifacts/benchmarks/benchmark_comparison_20261017_040343.json`  

---

## 1. Experimental Protocol & Boundary Definitions

To eliminate timing jitter and prevent biased comparisons:
1. **Timing Boundaries:** Timing explicitly isolates the **pure model forward pass**:
   $$T_{\text{total}} = T_{\text{prep}} + T_{\text{inference}} + T_{\text{post}}$$
   - Model loading time and graph compilation are **excluded** from latency measurements.
   - Preprocessing time (resizing, normalization, tensor packing) is measured separately.
   - Postprocessing time (coordinate un-scaling and visual binding) is measured separately.
2. **Warm-up Protocol:** Every benchmark executes **5 untimed warm-up cycles** to prime instruction caches and CPU frequency scaling governors before latency logging begins.
3. **Repetition Count:** **30 independent inference repetitions** are performed using the same lunar test tensor to provide statistically reliable distributions.
4. **Hardware Environment:**
   - Platform: Windows 11 (64-bit)
   - Host CPU: Intel64 Family 6 Model 126 Stepping 5 (4 logical cores, 2 physical cores)
   - System Memory: 11.77 GB RAM
   - Runtime: ONNX Runtime 1.31.0 (`CPUExecutionProvider`)

---

## 2. Mathematical Formulations

### 2.1 Storage Reduction
$$\text{Size Reduction (MB)} = \text{Size}_{\text{FP32}} - \text{Size}_{\text{INT8}}$$
$$\text{Percentage Size Reduction (\%)} = \left(\frac{\text{Size}_{\text{FP32}} - \text{Size}_{\text{INT8}}}{\text{Size}_{\text{FP32}}}\right) \times 100$$
*Measured:* $\left(\frac{14.29 - 4.10}{14.29}\right) \times 100 = \mathbf{71.31\%}$

### 2.2 Latency Speedup Ratio
$$\text{Speedup Ratio} = \frac{\text{Latency}_{\text{FP32}}}{\text{Latency}_{\text{INT8}}}$$
*Measured:* $\frac{16.37\text{ ms}}{236.57\text{ ms}} = \mathbf{0.069\times}$

### 2.3 Relative Latency Reduction
$$\text{Relative Latency Reduction (\%)} = \left(\frac{\text{Latency}_{\text{FP32}} - \text{Latency}_{\text{INT8}}}{\text{Latency}_{\text{FP32}}}\right) \times 100$$
*Measured:* $\left(\frac{16.37 - 236.57}{16.37}\right) \times 100 = \mathbf{-1345.1\%}$ (negative indicates latency increase on x86).

### 2.4 Percentage Points vs. Relative Percentage
- **Relative Percentage:** Measures fractional change relative to a baseline value (e.g., model size decreased by **71.3%** of its original size).
- **Percentage Points:** Measures arithmetic difference between two percentages (e.g., if precision changed from 1.33% to 1.21%, the delta is **-0.12 percentage points**, not -0.12%).

---

## 3. Empirical Benchmark Results Table

All figures below are extracted directly from the verified execution record:

| Metric | FP32 Baseline | INT8 Quantized | Unit | Scientific Interpretation |
|---|---|---|---|---|
| **Model Size** | 14.29 | 4.10 | MB | **71.31% storage compression** |
| **Minimum Latency** | 13.02 | 202.72 | ms | Best-case timing run |
| **Median Latency** | 16.37 | 236.57 | ms | Central tendency measurement |
| **Mean Latency** | 16.78 | 243.30 | ms | Average across 30 runs |
| **Standard Deviation** | 2.94 | 31.75 | ms | Latency stability index |
| **P90 Latency** | 21.03 | 278.23 | ms | 90th percentile worst-case |
| **P95 Latency** | 22.78 | 293.01 | ms | 95th percentile worst-case |
| **Maximum Latency** | 25.14 | 324.97 | ms | Extreme worst-case run |
| **Preprocessing Time**| 1.42 | 1.39 | ms | Image resizing and tensor norm |
| **Postprocessing Time**| 0.23 | 0.23 | ms | Coordinate un-scaling |
| **Throughput** | 59.58 | 4.11 | FPS | Frames evaluated per second |
| **Process Memory** | 461.30 | 461.34 | MB | Total host RAM footprint |
| **mAP@0.50** | 0.0002 | 0.0004 | — | 8-epoch proof-of-concept run |

---

## 4. Verification of Scientific Honesty

- **No Hardcoded Metrics:** Metrics in the dashboard are dynamically loaded from `artifacts/benchmarks/` and `artifacts/quantized/`.
- **Honest Detection Metrics:** The mAP@0.50 is transparently reported as proof-of-concept ($<0.01$) reflecting only 8 training epochs on 20 tiles without GPU augmentation.
- **Fair Hardware Comparison:** Both FP32 and INT8 models were evaluated on the identical CPU core using the identical test batch and ONNX Runtime environment.
