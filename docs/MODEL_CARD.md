# Model Card: SSDLite320 MobileNetV3-Large Crater Detector

**Model Name:** SSDLite320 MobileNetV3-Large (Lunar Crater Specialist)  
**Model Family:** Single Shot MultiBox Detector (SSD) with Lightweight Depthwise Convolutions  
**Task:** 2D Object Detection & Bounding-Box Regression  
**Target Domain:** Lunar Optical Surface Imagery (LRO NAC / Chandrayaan TMC/OHRC Analogs)  
**Artifact Paths:**
- Baseline PyTorch Checkpoint: `artifacts/checkpoints/crater_detector_baseline_best.pth` (42.99 MB)
- Exported FP32 ONNX Model: `artifacts/quantized/crater_detector_fp32.onnx` (14.29 MB)
- Quantized INT8 ONNX Model: `artifacts/quantized/crater_detector_int8.onnx` (4.10 MB)

---

## 1. Architectural Details

### 1.1 Backbone Network
- **Architecture:** MobileNetV3-Large.
- **Pretrained Weights:** ImageNet-1K (`MobileNet_V3_Large_Weights.DEFAULT` from torchvision).
- **Core Operations:** Inverted residual blocks with depthwise separable convolutions, hard-swish non-linearities, and squeeze-and-excitation (SE) attention modules.
- **Output Feature Maps:** Multi-scale feature pyramidal taps yielding feature maps with strides 16, 32, and beyond to capture craters of varying diameters.

### 1.2 Detection Head (SSDLite)
- **Design Philosophy:** Standard SSD uses heavy $3 \times 3$ standard convolutions for classification and regression predictors. SSDLite replaces these with **$3 \times 3$ depthwise convolutions followed by $1 \times 1$ pointwise convolutions**, reducing parameter count by ~75% and computational complexity by over 80%.
- **Output Classes:** 2 classes:
  - Class 0: `background`
  - Class 1: `crater` (target class)
- **Total Parameters:** ~3,710,000 trainable weights.

### 1.3 Prior Boxes & Anchor Generation
- **Aspect Ratios:** 1.0, 2.0, 0.5, 3.0, 0.33 (tuned to detect predominantly circular and elliptical craters formed by impact angles).
- **Scales:** Multi-scale default boxes covering small, medium, and large craters across the $320 \times 320$ receptive field.

---

## 2. Input and Output Tensor Specifications

### Input Tensor
- **Shape:** `[Batch_Size, 3, 320, 320]` (RGB format, channel-first).
- **Data Type:** `float32`.
- **Value Range:** Normalized by ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.

### Output Tensor Structure (Embedded ONNX Postprocessing)
The ONNX export incorporates the post-processing Non-Maximum Suppression (NMS) layer directly into the computational graph:
1. **`boxes`:** Shape `[N, 4]` — Float32 pixel coordinates in `[xmin, ymin, xmax, ymax]` convention relative to the 320×320 image frame.
2. **`scores`:** Shape `[N]` — Float32 confidence probabilities in `[0.0, 1.0]`.
3. **`labels`:** Shape `[N]` — Int64 class identifiers (1 for detected craters).

---

## 3. Training & Validation Protocol

- **Dataset:** 14 training tiles (~16,400 craters), 3 validation tiles (~3,400 craters).
- **Loss Function:**
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}} + \alpha \mathcal{L}_{\text{reg}}$$
  - Classification Loss $\mathcal{L}_{\text{cls}}$: Smooth L1 / Cross-Entropy with Hard Negative Mining (3:1 background-to-foreground ratio).
  - Localization Loss $\mathcal{L}_{\text{reg}}$: Smooth L1 loss on box offsets relative to default anchors.
- **Optimizer:** AdamW ($\text{LR} = 10^{-4}$, weight decay $10^{-4}$, cosine annealing schedule).
- **Epochs:** 8 epochs completed on local CPU execution provider.
- **Checkpoint Selection:** Best model selected by lowest validation loss and highest validation AP@0.50.

---

## 4. Hardware and Runtime Compatibility

- **Export Compatibility:** Standard TorchScript trace (`torch.onnx.export(..., dynamo=False)`) is mandatory because PyTorch 2.x `dynamo` / `torch.export` fails on SSD Non-Maximum Suppression dynamic boolean operations.
- **Inference Providers:**
  - `CPUExecutionProvider` (Intel/AMD x86, ARM64)
  - `CUDAExecutionProvider` (Optional on NVIDIA hardware)
- **Deployment Constraints:** Model runs fully within standard embedded spacecraft RAM limits (<50 MB peak memory during inference).
