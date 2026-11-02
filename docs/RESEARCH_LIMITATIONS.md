# Research Limitations & Future Technical Directions

**Project:** Micro-Quantized CNN for Onboard Crater Detection — Chandrayaan-Inspired Lunar Research  
**Document:** Scientific Honesty & Technical Limitations Declaration  

---

## 1. Dataset & Domain Coverage Limitations

1. **Geographic Localization:** The training and evaluation data is restricted to the floor of the **Von Kármán crater** in the Moon's South Pole–Aitken Basin. This terrain consists predominantly of relatively flat mare basalt plains. Highland terrains (such as the lunar farside highlands), which exhibit heavy crater saturation and degraded crater rims, are underrepresented.
2. **Solar Illumination Angle Sensitivity:** The LRO NAC imagery was captured under oblique solar illumination (incidence angles $55^\circ - 75^\circ$), where crater rims cast pronounced shadows and sun-facing walls exhibit sharp highlights. Under near-nadir illumination (sun overhead), cast shadows disappear, causing optical contrast to diminish and optical detection accuracy to drop sharply.
3. **Panchromatic Optical Monomodality:** The model processes optical reflectance only. Real planetary landing systems combine optical cameras with active sensors (LiDAR / Laser Altimeters) to measure physical depth independently of illumination.

---

## 2. Model Training & Accuracy Limitations

1. **Proof-of-Concept Training Extent:** The baseline detector was trained for **8 epochs on a multi-core CPU** using 14 training tiles (~16,400 craters). While sufficient to verify training convergence, loss reduction, and pipeline interoperability, achieving high mAP ($>0.70$) requires $100+$ epochs of multi-GPU training with extensive data augmentation (random flips, rotations, scale jitter, and synthetic illumination variation).
2. **Small Scale Crater Detection:** Craters with diameters smaller than 10 pixels in the resized $320 \times 320$ input tensor undergo spatial subsampling, making very small crater rims difficult to delineate accurately.

---

## 3. Quantization & Hardware Execution Limitations

1. **Dynamic vs. Static PTQ:** Our quantization pipeline uses **Dynamic Post-Training Quantization**, which dynamically computes activation scaling factors at runtime. While this eliminates the need for an offline calibration dataset, it introduces software-scaling overhead that increases inference latency on consumer x86 CPUs lacking AVX-VNNI extensions.
2. **Desktop Evaluation vs. Space Flight Hardware:** Benchmarks were performed on a commercial Intel x86 CPU. Although this empirically validates the **71.3% storage compression**, it does not measure real-time performance on spaceborne flight processors (e.g., SPARC LEON-4, Cobham UT699, or ARM Cortex-R52).

---

## 4. Spacecraft Integration Limitations

1. **Radiation Tolerance:** The software runs in user-space Python and ONNX Runtime; it does not implement Single Event Upset (SEU) fault detection, triple modular redundancy (TMR), or memory error-correcting code (ECC) monitoring required for deep-space flight software.
2. **Deterministic Latency:** High-level Python runtimes are subject to garbage collection pauses and OS thread scheduling jitter, precluding hard real-time compliance (such as ECSS-E-ST-40C space standards) without porting to C/C++ on an RTOS (like RTEMS).

---

## 5. Prioritized Future Work Roadmap

| Phase | Enhancement Description | Expected Technical Benefit |
|---|---|---|
| **Phase 1** | **Static PTQ with Calibration** | Precompute static activation scales using representative training tiles to eliminate runtime scaling latency. |
| **Phase 2** | **Multi-Modal Optical + DEM Fusion** | Concatenate Lunar Orbiter Laser Altimeter (LOLA) elevation maps as a 4th input channel to achieve illumination-independent detection. |
| **Phase 3** | **Global Dataset Expansion** | Expand training to the global LROC crater catalog (>1.3 million craters) using distributed GPU clusters. |
| **Phase 4** | **Hardware-in-the-Loop (HIL) Deployment** | Cross-compile the INT8 model for an ARM Cortex-R52 or Xilinx Space-Grade Zynq FPGA development board to measure real edge latency and milliwatt power consumption. |
