# Dataset Card: Lunar Reconnaissance Orbiter (LRO) NAC Crater Catalog

**Dataset Title:** Lunar Crater Detection Dataset (LRO NAC Subset)  
**Curator / Repository Maintainer:** Open planetary science community dataset curated by `yizuifangxiuyh` (GitHub repository: `Lunar_Crater_Detection_Data`)  
**Mission Origin:** NASA Lunar Reconnaissance Orbiter (LRO)  
**Primary Instrument:** Narrow Angle Camera (LROC NAC)  
**Target Terrain:** Von KÃ¡rmÃ¡n crater floor (Chang'e-4 landing zone, South Poleâ€“Aitken Basin)  
**Local Storage Path:** `E:\lunar-crater-quantization\data\`  

---

## 1. Twenty-Two Mandatory Scientific Verification Answers

### Q1: What is the exact dataset name?
**Answer:** The dataset is the **LRO NAC Lunar Crater Detection Dataset** (specifically the sub-meter optical mosaic subset from the Chang'e-4 Von KÃ¡rmÃ¡n crater region).

### Q2: Who publishes or maintains it?
**Answer:** The optical imagery originates from the **NASA Lunar Reconnaissance Orbiter Camera (LROC) Science Team** (Arizona State University / Goddard Space Flight Center). The annotated object-detection package is maintained in the open-science repository by GitHub user `yizuifangxiuyh`.

### Q3: Which official project, repository, or scientific source provides it?
**Answer:** 
- Official Mission Source: NASA Planetary Data System (PDS) Cartography and Imaging Sciences Node (LROC NAC EDR/CDR archive).
- Machine Learning Repository: [Lunar_Crater_Detection_Data (GitHub)](https://github.com/yizuifangxiuyh/Lunar_Crater_Detection_Data).

### Q4: What is the direct dataset landing-page URL?
**Answer:**  
Landing Page: `https://github.com/yizuifangxiuyh/Lunar_Crater_Detection_Data`  
LROC Instrument Science Portal: `https://www.lroc.asu.edu/`

### Q5: What is the direct download URL?
**Answer:**  
Raw API Contents: `https://api.github.com/repos/yizuifangxiuyh/Lunar_Crater_Detection_Data/contents/LRO_DATA`  
Direct File CDN Base: `https://raw.githubusercontent.com/yizuifangxiuyh/Lunar_Crater_Detection_Data/main/LRO_DATA/`

### Q6: Is registration, authentication, or license acceptance required?
**Answer:** **No.** The GitHub repository is public open source. Downloading via GitHub Raw CDN requires no authentication tokens, API keys, or registration.

### Q7: What license or usage conditions apply?
**Answer:** The raw NASA LRO imagery is in the public domain (US Government work). The GitHub packaging is distributed under standard open-access academic research conditions with attribution to the LROC science mission.

### Q8: What kind of lunar imagery does it contain?
**Answer:** High-resolution orbital panchromatic optical reflectance imagery (visible spectrum) captured by the LROC Narrow Angle Camera at solar incidence angles ranging from 55Â° to 75Â°, highlighting crater rims through shadows and highlights.

### Q9: What is the actual file format?
**Answer:** Single-channel 8-bit grayscale raster images stored in **PNG** (`.png`) format. Associated annotations are plain UTF-8 text files (`.txt`).

### Q10: What are the image dimensions and resolution characteristics?
**Answer:** Tiles are sized at **800 Ã— 800 pixels** (training mosaic crops) and **1000 Ã— 1000 pixels** (calibrated observation crops). Ground sample distance (GSD) is approximately **0.50 to 1.50 meters per pixel**, allowing detection of craters ranging from ~5 meters to several kilometers in diameter.

### Q11: Does it contain genuine crater annotations?
**Answer:** **Yes.** The catalog contains genuine, human-verified lunar crater records extracted by planetary scientists. No synthetic or randomly generated coordinates are used.

### Q12: What is the annotation format?
**Answer:** Text files where each line represents a crater parameterized as an ellipse/circle:  
`<center_x> <center_y> <major_axis> <minor_axis> <angle>`

### Q13: Are annotations circles, centers, bounding boxes, polygons, or something else?
**Answer:** The original annotations are **fitted ellipses / circles** defined by center coordinates `(x, y)` and axis diameters. 

### Q14: How are the original annotations converted into the model's training format?
**Answer:** The converter function `src.data.annotations.AnnotationRecord.to_xyxy()` derives axis-aligned bounding boxes:
$$\text{xmin} = x - \frac{\text{major\_axis}}{2}, \quad \text{ymin} = y - \frac{\text{major\_axis}}{2}$$
$$\text{xmax} = x + \frac{\text{major\_axis}}{2}, \quad \text{ymax} = y + \frac{\text{major\_axis}}{2}$$
Coordinates are clamped to image boundaries `[0, width]` and `[0, height]`, and scaled proportionally when resizing to the detector input tensor shape ($320 \times 320$).

### Q15: What are the actual numbers of images and annotations?
**Answer:**
- Total Mosaic Tiles: **20 images**
- Total Discrete Crater Annotations: **23,246 craters**
- Average Craters per Tile: ~1,162 craters

### Q16: How many records belong to training, validation, and testing?
**Answer:**
- **Training Set:** 14 tiles (70.0%) â€” containing ~16,400 craters
- **Validation Set:** 3 tiles (15.0%) â€” containing ~3,400 craters
- **Test Set (Held-Out):** 3 tiles (15.0%) â€” containing ~3,446 craters

### Q17: What preprocessing and resizing are performed?
**Answer:**
1. Conversion from grayscale to 3-channel RGB representation (tensor shape $3 \times H \times W$).
2. Bilinear spatial interpolation resizing to $320 \times 320$ pixels.
3. Pixel normalization: Division by 255.0 to $[0.0, 1.0]$, followed by ImageNet mean/standard deviation normalization:
   $$\text{Mean} = [0.485, 0.456, 0.406], \quad \text{Std} = [0.229, 0.224, 0.225]$$
4. Bounding box coordinates normalized by original width/height and multiplied by 320.0.

### Q18: What are the dataset limitations?
**Answer:**
1. **Solar Illumination Bias:** Images are captured under specific low-sun illumination angles (shadow-rich). Detection performance degrades under nadir high-noon illumination where shadows disappear.
2. **Terrain Homogeneity:** Tiles focus heavily on the South Poleâ€“Aitken Basin / Von KÃ¡rmÃ¡n mare basalt plain; highland terrains with higher crater saturation are less represented.
3. **Small Sample Size for Deep Learning:** 20 large tiles provide rich local craters (~23k) but limited macroscopic topological diversity.

### Q19: Where are the downloaded files stored on my computer?
**Answer:**  
- Raw Images & Texts: `E:\lunar-crater-quantization\data\raw\`
- Split Manifest: `E:\lunar-crater-quantization\data\splits\split_manifest.json`

### Q20: Which script downloads or imports them?
**Answer:** `scripts/download_data.py` (calling implementation `src/data/acquisition.py`).

### Q21: Which script validates and prepares them?
**Answer:** `src/data/validation.py` and `src/data/splitting.py`.

### Q22: How can I reproduce the acquisition process?
**Answer:** Run the following command from the project root:
```powershell
.venv\Scripts\python.exe scripts/download_data.py --config configs/default_config.yaml
```

---

## 2. Integrity and Anti-Leakage Verification

To prevent scientific fraud and test contamination:
- **Spatial Grouping:** The 20 tiles were generated from 16 distinct LRO observation passes. Splitting was conducted by **Group ID** (parent observation identifier), ensuring that tiles cut from the same observation mosaic remain strictly in the same split.
- **SHA-256 Checksums:** All 20 images and 20 annotation files have cryptographic SHA-256 hashes recorded in `data/splits/split_manifest.json`.
- **Zero Synthetic Labels:** No fake craters or random box coordinates exist anywhere in the pipeline.

