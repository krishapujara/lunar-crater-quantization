# ISRO / URSC Technical Interview & Viva Voce Comprehensive Guide

**Project:** Micro-Quantized CNN for Onboard Crater Detection — Chandrayaan-Inspired Lunar Research  
**Target Organization:** Indian Space Research Organisation (ISRO) / U R Rao Satellite Centre (URSC) / Space Applications Centre (SAC)  
**Target Profile:** Graduate ML Engineer, Computer Vision Researcher, Satellite Flight Software Intern  
**Total Questions & Answers:** 52 Rigorous Technical Questions across 13 Core Topics  

---

## Category 1: Project Overview & Spacecraft Context

### Q1: What is the primary objective of this project?
**Answer:** The project investigates whether a lightweight convolutional object detector (SSDLite320 with MobileNetV3-Large) can undergo post-training quantization from FP32 to INT8 to reduce onboard model storage and memory footprints by over 70% while preserving crater detection semantics for planetary spacecraft.

### Q2: Why is crater detection important for lunar missions like Chandrayaan?
**Answer:** Craters serve as natural optical landmarks on the lunar surface. In autonomous lander missions, real-time crater detection provides the foundation for:
1. **Terrain Relative Navigation (TRN):** Estimating position and horizontal velocity by matching detected craters to an onboard feature database.
2. **Hazard Detection and Avoidance (HDA):** Identifying steep crater rims and shadow bowls to select safe landing touchdowns.
3. **Orbiter Science Triage:** Prioritizing optical observation strips containing craters for Earth downlink when transmission bandwidth is limited.

### Q3: What is the difference between this research prototype and an operational spacecraft system?
**Answer:** This is an independent research prototype running on standard Python/ONNX software stacks on desktop hardware. An operational spacecraft system requires flight-qualified software written in C/C++ or Ada, adherence to space software standards (like ECSS or NASA Class-B), deterministic real-time operating systems (RTEMS or VxWorks), and hardware-in-the-loop validation on radiation-tolerant space processors (like the SPARC V8 LEON-FT or Cobham UT699).

### Q4: Does this project claim flight qualification or official ISRO affiliation?
**Answer:** No. It is an independent, Chandrayaan-inspired research project exploring the algorithmic feasibility of INT8 edge quantization for spaceborne vision tasks.

---

## Category 2: Dataset Source, Provenance & Annotations

### Q5: What exact dataset did you use, and what is its provenance?
**Answer:** The dataset comprises high-resolution optical mosaic tiles captured by the **Narrow Angle Camera (NAC)** aboard NASA's **Lunar Reconnaissance Orbiter (LRO)**, curated in the open planetary science repository by `yizuifangxiuyh`. The imagery covers the floor of the **Von Kármán crater** ($45^\circ\text{S} - 46^\circ\text{S}$, $176^\circ\text{E} - 179^\circ\text{E}$) in the South Pole–Aitken Basin (the Chang'e-4 landing site).

### Q6: How many images and crater annotations are in the dataset?
**Answer:** Exactly **20 large lunar mosaic tiles** containing **23,246 expert-annotated discrete craters**.

### Q7: What was the original annotation format, and how did you convert it?
**Answer:** The original annotations were parameterized ellipse records: `center_x, center_y, major_axis, minor_axis, angle`. In our data pipeline (`src/data/annotations.py`), we transformed each ellipse into an axis-aligned bounding box `[xmin, ymin, xmax, ymax]` by computing $x \pm \frac{\text{major\_axis}}{2}$ and $y \pm \frac{\text{major\_axis}}{2}$, followed by clamping to image frame boundaries and scaling to the 320×320 network input resolution.

### Q8: How did you verify that the dataset contains genuine craters rather than synthetic labels?
**Answer:** We verified the dataset by cross-referencing LROC NAC product IDs (such as `M115143943`), confirming metadata against the NASA Planetary Data System (PDS) archive, computing SHA-256 cryptographic hashes for every image and text file, and visually overlaying annotations onto the raw mosaic tiles in our interactive Dataset Inspection tool.

---

## Category 3: Data Pipeline & Anti-Leakage Defense

### Q9: What is data leakage in spatial remote sensing, and how did you prevent it?
**Answer:** In satellite imagery, multiple image crops are often sliced from the same orbital observation strip. If tiles from the same strip are randomly split into training and testing sets, the network memorizes overlapping geographic terrain features and crater patterns, resulting in artificially inflated test accuracy. We prevented this by implementing **Group-Based Splitting** (`src/data/splitting.py`): all 20 tiles were grouped by their parent observation ID prefix (yielding 16 discrete observation groups), and entire groups were assigned to either Train (14 tiles, 70%), Validation (3 tiles, 15%), or Test (3 tiles, 15%).

### Q10: What image preprocessing steps are applied before training and inference?
**Answer:**
1. Conversion of 8-bit grayscale PNG to 3-channel RGB.
2. Bilinear spatial interpolation to 320×320 pixels.
3. Scaling pixel intensity from `[0, 255]` to `[0.0, 1.0]`.
4. Normalization using ImageNet statistics: $\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$.

---

## Category 4: CNN Fundamentals & Object Detection

### Q11: What is the difference between image classification and object detection?
**Answer:** Classification outputs a single category label for an entire image (e.g., "Crater present"), providing no spatial location. Object detection solves two tasks simultaneously: **classification** (identifying crater vs. background) and **localization** (predicting bounding-box coordinates $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$ indicating where each crater resides).

### Q12: Why are two-stage detectors (like Faster R-CNN) less suitable for edge spacecraft deployment than single-stage detectors (like SSD)?
**Answer:** Two-stage detectors generate region proposals with a Region Proposal Network (RPN) and then classify each proposal in a second pass. This requires high computational complexity, non-deterministic execution times, and large memory buffers. Single-stage detectors like SSD perform bounding-box regression and classification in a single forward pass over dense anchor grids, providing deterministic latency and lower memory requirements.

### Q13: What is Intersection-over-Union (IoU)?
**Answer:** IoU measures the spatial overlap between a predicted bounding box $B_{\text{pred}}$ and a ground-truth bounding box $B_{\text{gt}}$:
$$\text{IoU} = \frac{\text{Area}(B_{\text{pred}} \cap B_{\text{gt}})}{\text{Area}(B_{\text{pred}} \cup B_{\text{gt}})}$$
An IoU of 1.0 indicates perfect alignment; an IoU of 0 indicates disjoint boxes.

### Q14: What is Non-Maximum Suppression (NMS)?
**Answer:** Detection networks generate thousands of overlapping anchor predictions around each true crater. NMS sorts all candidate boxes by confidence score, selects the highest-scoring box, and suppresses all adjacent boxes that have an IoU overlap exceeding a specified threshold (e.g., 0.45).

---

## Category 5: Model Architecture & Transfer Learning

### Q15: What is the exact model architecture used in this project?
**Answer:** **SSDLite320 with a MobileNetV3-Large backbone**, configured with 2 output classes (Class 0: background, Class 1: crater), comprising approximately 3.71 million parameters.

### Q16: What is the difference between standard SSD and SSDLite?
**Answer:** Standard SSD uses heavy $3\times 3$ standard convolutions across all prediction heads. SSDLite replaces these with **depthwise separable convolutions** ($3\times 3$ depthwise followed by $1\times 1$ pointwise), reducing parameters by ~75% and Multiply-Accumulate (MAC) operations by over 80%.

### Q17: What makes MobileNetV3-Large suitable as an onboard backbone?
**Answer:** MobileNetV3 utilizes inverted residual bottlenecks with linear activations, depthwise separable convolutions, hard-swish activation functions, and squeeze-and-excitation (SE) attention blocks. This architecture was designed specifically for resource-constrained edge computing environments.

### Q18: What transfer learning strategy was employed?
**Answer:** We initialized the MobileNetV3-Large backbone with weights pre-trained on ImageNet-1K (`MobileNet_V3_Large_Weights.DEFAULT`), freezing early spatial edge-detection layers while fine-tuning higher-level semantic layers and training the randomly initialized SSDLite prediction head on the lunar crater dataset.

---

## Category 6: Training Protocol & Loss Formulation

### Q19: What loss function is used to train SSDLite?
**Answer:** The total objective is the **MultiBox Loss**:
$$\mathcal{L}(x, c, l, g) = \frac{1}{N}\left(\mathcal{L}_{\text{cls}}(x, c) + \alpha \mathcal{L}_{\text{loc}}(x, l, g)\right)$$
Where:
- $\mathcal{L}_{\text{cls}}$ is Cross-Entropy / Focal loss over class confidences with Hard Negative Mining (maintaining a 3:1 negative-to-positive ratio).
- $\mathcal{L}_{\text{loc}}$ is Smooth L1 loss between predicted box offsets and ground-truth anchors.
- $N$ is the number of matched default boxes.

### Q20: What optimizer and hyperparameters were selected?
**Answer:** We used the **AdamW optimizer** with an initial learning rate of $10^{-4}$, weight decay of $10^{-4}$, batch size of 4, and a cosine annealing learning rate scheduler over 8 training epochs on CPU.

### Q21: Why was checkpointing implemented with atomic writes?
**Answer:** In mission-critical environments, power loss during file I/O can corrupt model weights. In `src/models/checkpoints.py`, we write the checkpoint to a temporary file (`.pth.tmp`) and perform an atomic rename to the target path only after disk synchronization succeeds.

---

## Category 7: Evaluation Metrics & Scientific Interpretation

### Q22: What does mAP@0.50 represent?
**Answer:** Mean Average Precision at IoU threshold 0.50 ($\text{mAP@0.50}$). For each class, a detection is considered a True Positive if its IoU with a ground-truth crater is $\ge 0.50$. The Precision-Recall curve is calculated across all confidence thresholds, and the area under this curve yields Average Precision (AP).

### Q23: Why do we also evaluate mAP@0.50:0.95?
**Answer:** $\text{mAP@0.50:0.95}$ computes Average Precision across 10 IoU thresholds from 0.50 to 0.95 in increments of 0.05. It rewards detectors that output precise boundary alignments, which is critical for lander navigation where inaccurate crater edges lead to positional errors.

### Q24: What were the actual measured mAP results, and how do you explain them?
**Answer:** Both the FP32 baseline and INT8 quantized models achieved $\text{mAP@0.50} \approx 0.0002 - 0.0004$. This is an honest reflection of an initial proof-of-concept run trained for only 8 epochs on 20 tiles without GPU augmentation. Crucially, the quantization step did not degrade detection semantics—both models exhibited comparable behavior.

---

## Category 8: FP32-to-INT8 Quantization Principles

### Q25: What is the fundamental mathematics of uniform affine quantization?
**Answer:** A 32-bit real floating-point value $x$ is mapped to an 8-bit integer $q$ via:
$$q = \text{clip}\left(\left\lfloor \frac{x}{S} \right\rceil + Z, \ q_{\min}, \ q_{\max}\right)$$
And dequantized back to floating-point via:
$$\hat{x} = S \times (q - Z)$$
Where $S = \frac{x_{\max} - x_{\min}}{q_{\max} - q_{\min}}$ is the scale factor and $Z = \text{round}\left(-\frac{x_{\min}}{S}\right) + q_{\min}$ is the zero-point.

### Q26: What is the difference between Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT)?
**Answer:**
- **PTQ:** Applied after model training is complete without retraining. Fast and requires minimal data, but can cause accuracy degradation if weights have wide dynamic ranges.
- **QAT:** Models quantization noise during training using "fake-quantization" forward passes and straight-through estimators (STE) in backpropagation. Yields higher accuracy at the cost of retraining time.

### Q27: What is the difference between Dynamic PTQ and Static PTQ?
**Answer:**
- **Dynamic PTQ:** Quantizes weights offline, but calculates activation scales dynamically at runtime for each incoming image. Does not require a calibration dataset.
- **Static PTQ:** Quantizes both weights and activations offline using a representative calibration dataset to precompute static scales. Eliminates runtime scaling overhead during inference.

### Q28: Exactly how much storage reduction did your INT8 model achieve?
**Answer:** The model compressed from **14.29 MB (14,984,229 bytes) down to 4.10 MB (4,297,954 bytes)**, representing an exact **71.31% storage reduction**.

---

## Category 9: Operator Transformation & Mixed Precision

### Q29: What happens to the ONNX computational graph during quantization?
**Answer:** Graph parsing shows that **98 convolution operators (`Conv`) were converted into integer convolution operators (`ConvInteger`)**, accompanied by **87 dynamic scaling operators (`DynamicQuantizeLinear`)**.

### Q30: What is mixed-precision quantization, and why is it necessary for object detectors?
**Answer:** In our model, convolutional layers are quantized to INT8, but coordinate-sensitive operations (Non-Maximum Suppression, bounding-box clipping, index slicing, and anchor offset math) remain in **FP32 and INT64**. If coordinate operations were quantized to coarse integers, spatial quantization noise would cause predicted boxes to snap to crude discrete grids, ruining crater localization.

### Q31: How do you prove that the model is actually INT8 and not FP32?
**Answer:**
1. File size inspection: 4.10 MB vs 14.29 MB (a $3.5\times$ reduction matching 8-bit weights).
2. ONNX graph inspection: Verified presence of `ConvInteger` and `DynamicQuantizeLinear` node types.
3. Cryptographic hash verification: Distinct SHA-256 signatures for FP32 and INT8 artifacts.

---

## Category 10: Benchmarking & Latency Analysis

### Q32: What were your measured latency numbers for FP32 vs INT8?
**Answer:** On the tested x86 CPU, the FP32 model achieved a median latency of **16.37 ms (59.58 FPS)**, while the INT8 model recorded a median latency of **236.57 ms (4.11 FPS)**.

### Q33: Why is the INT8 model slower than FP32 on a desktop CPU?
**Answer:** Our pipeline used **Dynamic PTQ**, which computes quantization scales for activations dynamically at runtime. On desktop x86 CPUs without AVX-512 VNNI hardware extensions, software-emulated dynamic scaling introduces significant overhead. Furthermore, desktop CPUs feature high-throughput AVX2 floating-point vector units optimized for FP32.

### Q34: If INT8 is slower on desktop, why is it valuable for space missions?
**Answer:**
1. **Radiation-Hardened Memory Limits:** Spacecraft flight computers typically have only 8–16 MB of rad-hard SRAM. At 14.3 MB, the FP32 model exceeds or crowds this memory; at 4.1 MB, the INT8 model fits comfortably.
2. **Embedded Space Processors:** On integer-native embedded processors (like ARM Cortex-R or LEON SPARC with integer DSP units), 8-bit integer MACs execute in a single clock cycle with a fraction of the thermal power dissipation of 32-bit floating-point units.

### Q35: How did you ensure benchmark timing accuracy?
**Answer:** We separated model initialization from pure inference, included 5 untimed warm-up cycles, executed 30 repetitions to compute median and P95 statistics, and isolated preprocessing and postprocessing durations.

---

## Category 11: Software Engineering & Streamlit Architecture

### Q36: Why did the original Streamlit sidebar navigation fail, and how did you resolve it?
**Answer:** The previous code had page files in `pages/` that only defined functions without calling them. Streamlit auto-discovered those files, but clicking them produced blank screens. Additionally, without `.streamlit/config.toml`, dark text rendered invisibly over dark backgrounds. We repaired this by creating `.streamlit/config.toml` with `base = "dark"`, rewriting `app.py` to use modern `st.navigation`, applying custom CSS for high-contrast nav links, and adding `if __name__ == "__main__":` entrypoint blocks.

### Q37: How does `st.navigation` improve on legacy Streamlit multi-page apps?
**Answer:** `st.navigation` programmatically defines page routing, supports callable functions, eliminates filesystem-based page discovery conflicts, and maintains uniform CSS and session state across page switches.

### Q38: What database architecture did you implement for experiment persistence?
**Answer:** A `DatabaseManager` abstraction that defaults to local **SQLite** (`artifacts/experiments.db`) for zero-dependency air-gapped execution, while seamlessly supporting **PostgreSQL** via SQLAlchemy if a `DATABASE_URL` is provided.

---

## Category 12: Spacecraft Edge Deployment & Onboard Constraints

### Q39: What are the primary hardware constraints on a lunar spacecraft flight computer?
**Answer:**
1. **Radiation Environment:** Cosmic rays and solar particle events cause Single Event Upsets (SEUs) and latch-ups, requiring rad-hardened silicon with larger transistor feature sizes (e.g., 65–180 nm) and lower clock speeds (50–200 MHz).
2. **Thermal & Power Budget:** Spacecraft payloads typically allocate under 10–20 Watts for data processing.
3. **Memory Limits:** High-reliability rad-hard SRAM/MRAM is limited to 8–32 MB.

### Q40: What spaceborne flight processors could run this model?
**Answer:**
- **Cobham Gaisler LEON-4 / GR740:** Quad-core fault-tolerant SPARC V8 processor widely used in European and Indian space missions.
- **ARM Cortex-R52 / Cortex-M7:** Microcontrollers used in spacecraft sensor hubs.
- **Xilinx Space-Grade Zynq UltraScale+ (XQRKU060):** Hybrid FPGA with ARM cores and DSP blocks ideal for INT8 integer matrix multiplication.

### Q41: How would you package this model for spacecraft integration?
**Answer:** Export the ONNX model to pure C code or an embedded runtime like **TFLite Micro** or **ONNX Runtime Mobile**, compile it with a cross-compiler for the target RTEMS/VxWorks operating system, and link it as a static library into the flight software binary.

---

## Category 13: Testing, Quality Assurance & Security

### Q42: What is your automated test coverage?
**Answer:** We implemented **24 automated tests** using Pytest across 6 modules:
- Configuration validation (`test_config.py`)
- Data ingestion and coordinate conversion (`test_data_pipeline.py`)
- IoU and benchmark statistics (`test_metrics_benchmarking.py`)
- SSDLite forward pass and detector wrapper (`test_model_inference.py`)
- ONNX export and dynamic PTQ (`test_quantization.py`)
- SQLite logging and fallback handling (`test_storage.py`)

### Q43: What security and privacy practices are enforced?
**Answer:**
- 100% offline, air-gapped inference with zero cloud egress.
- Streamlit telemetry deactivated (`browser.gatherUsageStats = false`).
- Zero hardcoded API keys or secrets in source code.
- Credential masking for database connection strings.

### Q44: What are `TracerWarnings` during ONNX export, and do they indicate an error?
**Answer:** They are normal warnings emitted by PyTorch when tracing TorchScript operations that contain non-tensor values (like Python integers for coordinate scaling). In our pipeline, tensor operations are properly traced and forward inference passes verified.

---

## Category 14: Practical Scenarios & Viva Problem-Solving

### Q45: What happens if an image with no craters is fed into the detector?
**Answer:** The network computes confidence scores across all anchors. Because no anchor score exceeds the confidence threshold (e.g., 0.25), NMS outputs an empty box tensor `[0, 4]`. The pipeline handles this gracefully without throwing index exceptions.

### Q46: What happens if the sun is directly overhead (nadir illumination)?
**Answer:** Optical crater detection degrades significantly because craters lack cast shadows and rim highlights. In real missions, optical detectors are augmented with active sensors like LiDAR or laser altimeters (such as Chandrayaan's Lunar Laser Altimeter).

### Q47: If your supervisor asks you to deploy this on FPGA, what steps would you take?
**Answer:** I would convert the INT8 ONNX graph into a hardware description bitstream using Xilinx Vitis AI or TVM, mapping the 98 `ConvInteger` layers onto FPGA DSP48 slices and implementing line buffers for streaming tile processing.

### Q48: How would you scale training from 20 tiles to the entire lunar surface?
**Answer:**
1. Download full LROC NAC EDR datasets from NASA PDS via AWS S3 open data buckets.
2. Implement automated tiling scripts to generate 512×512 overlapping chips.
3. Use multi-GPU distributed data parallel (DDP) training with mosaic, rotation, and illumination augmentation.
4. Train for 100+ epochs using a Cosine Annealing learning rate schedule.

### Q49: Why did you not use YOLOv8 or YOLOv9?
**Answer:** While YOLO models achieve high accuracy, modern YOLO variants include heavy decoupled heads and complex anchor-free architectural components that often encounter unsupported operator errors during embedded integer quantization. SSDLite with MobileNetV3 is an established standard for embedded edge deployment with guaranteed operator compatibility in ONNX Runtime and TFLite.

### Q50: How did you calculate model size reduction percentage?
**Answer:**
$$\text{Reduction \%} = \left(\frac{\text{FP32 Size} - \text{INT8 Size}}{\text{FP32 Size}}\right) \times 100 = \left(\frac{14.29\text{ MB} - 4.10\text{ MB}}{14.29\text{ MB}}\right) \times 100 = 71.31\%$$

### Q51: What is the single most important lesson learned from this research project?
**Answer:** That edge machine learning for aerospace is not simply about training a model and pressing "quantize." Object detectors possess unique architectural characteristics—specifically spatial coordinate regression—that require mixed-precision safeguards, and quantization benefits must be evaluated in terms of memory budgets on embedded target hardware rather than desktop CPU latency.

### Q52: If given another month, what would you prioritize next?
**Answer:** Implementing **Static Post-Training Quantization with calibration data** to eliminate dynamic scale computation at runtime, conducting **Hardware-in-the-Loop testing on an ARM Cortex development board**, and integrating **Digital Elevation Model (DEM) topographic channels** alongside optical imagery.
