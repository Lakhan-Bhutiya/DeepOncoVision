#  DeepOncoVision
### *Because one doctor's opinion is good, but a doctor + an AI trained on 220,000 images + 40,000 hospital records is better*

---

> We built an AI that looks at cancer tissue images AND reads patient medical notes at the same time, then combines both to predict cancer. It achieved **90% accuracy**. No, we didn't cure cancer. But we're having a good time trying.

---

##  What Even Is This?

Imagine you're a doctor. A patient walks in. You could:

**Option A:** Look at their scan, shrug, and say "yeah that looks fine probably"

**Option B:** Look at their scan AND read their full medical history, smoking habits, symptoms, and family history before making a decision

DeepOncoVision is Option B. But for AI. Built in Python. By people who spent way too long debugging a missing `/` in a file path.

---

##  Project Structure

```
DeepOncoVision/
│
├── 📂 big_data/               # It's in the name. Big. Data.
│
├── 📂 Books/                  #all the .ipynb bookes if you want to really run this it will take like decads to run
                               # jk it will run in 8 hours minimum 
│
├── 📂 cancer_data/            # 220,025 tissue images
│                               # Each one 96×96 pixels of pure anxiety
│
├── 📂 mimic_demo/             # Real hospital records from MIMIC-III
│                               # 40,000 patients who consented to science
│
├── 📂 Models/                 # Where the magic lives
│   ├── best_model.pth         # CNN: The eye doctor (86% accurate)
│   ├── best_bert_model.pth    # BERT: The note reader (93% accurate)
│   └── best_fusion_model.pth  # The combined genius (90% accurate)
│
├── 📂 train_split/            # 80% of data (for learning)
│                               # Like studying for an exam
│
├── 🖼️ grid_results.png        # 9 predictions at once, green=correct red=wrong
├── 🖼️ OUTPUT_FINAL_...        # The moment of truth
└── 🖼️ prediction_result...    # Individual predictions with confidence bars
```

---

## Architecture — The Full Picture
 
DeepOncoVision follows a **multimodal pipeline** — two separate AI brains that each learn something different, then combine their knowledge for a final verdict.
 
```
 INPUT LAYER
 ═══════════════════════════════════════════════════════════════
 
     CT/MRI Tissue Image                Clinical Text Record
   (Histopathologic dataset)           (MIMIC-III hospital notes)
   220,025 images · 96×96px            40,000+ discharge summaries
          │                                       │
          │                                       │
          ▼                                       ▼
 
 PREPROCESSING LAYER
 ═══════════════════════════════════════════════════════════════
 
    Image Pipeline                     Text Pipeline
   ┌─────────────────────┐             ┌─────────────────────┐
   │ • Resize to 96×96   │             │ • Discharge summary │
   │ • Random H/V flip   │             │   extraction        │
   │ • Rotation ±15°     │             │ • ICD-9 code lookup │
   │ • Color jitter      │             │   (140–239 = cancer)│
   │ • Normalize         │             │ • Truncate to 512   │
   │   (ImageNet mean)   │             │   characters        │
   └─────────────────────┘             └─────────────────────┘
          │                                       │
          │                                       │
          ▼                                       ▼
 
 FEATURE EXTRACTION LAYER
 ═══════════════════════════════════════════════════════════════
 
    CNN Image Encoder                   BERT Text Encoder
   ┌─────────────────────┐             ┌─────────────────────┐
   │ ResNet18             │             │ bert-base-uncased   │
   │ (pretrained on      │             │ (fine-tuned on      │
   │  ImageNet)          │             │  MIMIC-III)         │
   │                     │             │                     │
   │ Frozen layers +     │             │ [CLS] token →       │
   │ custom head:        │             │ Dropout(0.3) →      │
   │ Linear(512→256) →   │             │ Linear(768→256) →   │
   │ ReLU →              │             │ ReLU →              │
   │ Dropout(0.4) →      │             │ Dropout(0.3) →      │
   │ Linear(256→2)       │             │ Linear(256→2)       │
   │                     │             │                     │
   │ Output:             │             │ Output:             │
   │ 512-dim feature     │             │ 768-dim feature     │
   │ vector              │             │ vector              │
   │                     │             │                     │
   │ Accuracy: 86%       │             │ Accuracy: 93%       │
   └─────────────────────┘             └─────────────────────┘
          │                                       │
          └───────────────┬───────────────────────┘
                          │
                          ▼
 
 MULTIMODAL FUSION LAYER
 ═══════════════════════════════════════════════════════════════
 
              ┌────────────────────────────────┐
              │    Concatenate features        │
              │    512 + 768 = 1280-dim        │
              │                                │
              │    Fusion MLP:                 │
              │    Linear(1280 → 512)          │
              │    ReLU + Dropout(0.3)         │
              │    Linear(512 → 128)           │
              │    ReLU + Dropout(0.2)         │
              │    Linear(128 → 2)             │
              │                                │
              │    Both CNN and BERT frozen    │
              │    Only fusion head trains     │
              └────────────────────────────────┘
                          │
                          ▼
 
 OUTPUT LAYER
 ═══════════════════════════════════════════════════════════════
 
              ┌────────────────────────────────┐
              │   Cancer Classification        │
              │                                │
              │   Normal:  57.6% ──────────    │
              │   Cancer:  42.4% ───────        │
              │                                │
              │   Final verdict + confidence   │
              │   score 0–100%                 │
              └────────────────────────────────┘
                          │
                          ▼
 
 EXPLAINABILITY LAYER
 ═══════════════════════════════════════════════════════════════
 
    Grad-CAM (Images)                   Attention (Text)
   ┌─────────────────────┐             ┌─────────────────────┐
   │ Highlights which    │             │ Shows which words   │
   │ pixels in the scan  │             │ in the clinical     │
   │ the CNN focused on  │             │ note influenced     │
   │ when it decided     │             │ BERT's prediction   │
   │                     │             │                     │
   │ Output: heatmap     │             │ Output: highlighted │
   │ overlaid on image   │             │ text tokens         │
   └─────────────────────┘             └─────────────────────┘
                          │
                          ▼
 
 RESULTS DISPLAY
 ═══════════════════════════════════════════════════════════════
 
              ┌────────────────────────────────┐
              │  Image | CNN bar | BERT bar    │
              │        | Fusion bar            │
              │  Clinical note shown below     │
              │  True label vs prediction      │
              │  ✅ CORRECT  /  ❌ WRONG        │
              └────────────────────────────────┘
```

---
 
## 🔄 Data Flow — Step by Step
 
| Step | What happens | Input | Output |
|------|-------------|-------|--------|
| 1 | Image loaded from disk | `.tif` file path | PIL Image |
| 2 | Image transformed | PIL Image | Tensor `[3, 96, 96]` |
| 3 | CNN feature extraction | Image tensor | 512-dim vector |
| 4 | Clinical note tokenized | Raw text string | `input_ids` + `attention_mask` |
| 5 | BERT feature extraction | Tokenized text | 768-dim vector |
| 6 | Features concatenated | 512 + 768 vectors | 1280-dim vector |
| 7 | Fusion MLP forward pass | 1280-dim vector | 2 logits |
| 8 | Softmax applied | 2 logits | 2 probabilities (sum to 100%) |
| 9 | Argmax gives prediction | 2 probabilities | `0=Normal` or `1=Cancer` |
 
---
##  Dataset Decision — Why Not LIDC-IDRI?
 
The original project proposal referenced LIDC-IDRI (CT/MRI scans). Here is why we used the Histopathologic Cancer Detection dataset instead:
 
> LIDC-IDRI was evaluated as a potential imaging dataset, however it was not feasible for this project for three key reasons. First, the full dataset requires approximately 125–150 GB of storage which exceeded our available compute environment. Second, it contains only 1,018 CT scan cases in raw DICOM format requiring complex 3D preprocessing pipelines before any training can begin. In contrast, the Histopathologic Cancer Detection dataset provided 220,025 clean labeled image patches with unambiguous binary labels, making it significantly more suitable for demonstrating our multimodal fusion architecture.

---

## Datasets Used

###  Histopathologic Cancer Detection (Kaggle)
- **220,025** tissue images at 96×96 pixels
- Binary labels: `0` = normal tissue, `1` = cancer tissue
- Stored at: `/home/zeus/content/cancer_data/train/`
- Fun fact: We spent 3 days debugging a path error. The fix was adding one `/`

###  MIMIC-III Clinical Database (v1.4)
- Real de-identified hospital records from Beth Israel Deaconess Medical Center
- We used `NOTEEVENTS.csv` (discharge summaries) + `DIAGNOSES_ICD.csv`
- Cancer label: any ICD-9 code between **140–239** = cancer
- Fun fact: We first tried synthetic notes. BERT got 100% accuracy by learning that "5-year-old" = normal. It was cheating. We caught it.

---

## How It Actually Works

### Step 1: The CNN reads the image

```
Tissue image (96×96 pixels)
         ↓
ResNet18 (pre-trained on ImageNet, fine-tuned on our data)
         ↓
512 numbers describing "what the image looks like"
         ↓
"This looks 87% like cancer tissue"
```

### Step 2: BERT reads the clinical note

```
"Patient is a 67-year-old male, 45 pack-year smoking history,
 CT shows 2.8cm spiculated nodule, hemoptysis for 4 months..."
         ↓
bert-base-uncased (fine-tuned on MIMIC-III)
         ↓
768 numbers describing "what the patient's situation is"
         ↓
"This patient has 93% cancer risk indicators"
```

### Step 3: Fusion decides

```
512 (image) + 768 (text) = 1280 numbers
         ↓
Small MLP: Linear(1280→512)→ReLU→Linear(512→128)→ReLU→Linear(128→2)
         ↓
"Final verdict: CANCER — 90% confidence"
```

---

##  Quick Start

### Prerequisites

```bash
pip install torch torchvision transformers scikit-learn pandas matplotlib pillow tqdm
```

### Running predictions

```python
# Load all models (run once)
# See: prediction_notebook.ipynb — Cell 1 & 2

# Pick any image by index (0 to 44,004)
IMAGE_INDEX = 42
CLINICAL_NOTE = None  # or write your own

result = predict(IMAGE_INDEX, CLINICAL_NOTE)
show_result(result)
# Output: image + 3 bar charts + clinical note
```

---

##  Results

| Model | What it uses | Accuracy |
|-------|-------------|----------|
| CNN alone | Tissue image only | **86%** |
| BERT alone | Clinical notes only | **93%** |
| Fusion (combined) | Image + Text | **90%** |

> *"Wait, why is Fusion lower than BERT alone?"*
> Great question. BERT trained on real MIMIC-III notes with real ICD-9 cancer codes — that's a strong signal. The Fusion model is learning to balance both. With more data alignment between the two datasets, fusion would win. This is a research project, not WebMD.

---

##  Bugs WE FACED

A love letter to every error message we received:

| Error | What it meant | How we fixed it |
|-------|--------------|-----------------|
| `FileNotFoundError: cancer-detection/train7882da...tif` | Missing `/` between folder and filename | Added one single `/` |
| `Combined model always predicts NORMAL` | Combiner learned wrong weights from fake notes | Replaced combiner with weighted average |
| BERT getting 100% accuracy | It was cheating — memorised patient ages from synthetic notes | Replaced with real MIMIC-III data |
| GPU at 9% utilisation | CPU loading bottleneck — GPU was waiting for data | batch_size=256, num_workers=4, pin_memory=True |

---

##  Notebooks

| Notebook | What it does |
|----------|-------------|
| `CNN_MODEL.ipynb` | Train ResNet18 on tissue images |
| `BERT_MODEL.ipynb` | Fine-tune BERT on MIMIC-III clinical notes |
| `FUSION_MODEL.ipynb` | Combine CNN + BERT via fusion MLP |
| `OUTPUT_FINAL.ipynb` | Interactive predictions — give index, get diagnosis |

---

## Clinical Note Examples

Want to test the model with your own clinical text? Here are the kinds of notes that push BERT toward each prediction:

**High risk (cancer indicators):**
> *"67-year-old male, 45 pack-year smoking history. CT shows 2.8cm spiculated nodule right upper lobe. Hemoptysis for 4 months. Biopsy confirmed adenocarcinoma."*

**Low risk (normal indicators):**
> *"24-year-old female, non-smoker, no family history. Admitted for appendicitis. Chest clear. No lymphadenopathy. No malignancy identified."*

---

## Tech Stack

```
Language    : Python 3.12
Framework   : PyTorch + HuggingFace Transformers
CNN         : torchvision ResNet18
BERT        : bert-base-uncased
Training    : Mixed precision (AMP) — torch.cuda.amp
Environment : Lightning AI studio (GPU T4)
Datasets    : Kaggle Histopathologic + MIMIC-III v1.4
Visualisation: matplotlib
```

---

##  Environment Setup

This project was built and trained on **Lightning AI** with a T4 GPU.

Key paths on the training machine:
```

Images  : /home/zeus/content/cancer_data/train/
CSV     : /home/zeus/content/cancer_data/train_labels.csv
Models  : /Models/CNN_MODEL.pth
          /Models/BERT_MODEL.pth
          /Models/FUSION_MODEL.pth

```

---

##  Limitations

- The tissue images are from a histopathology dataset, not actual CT/MRI scans. Real CT scans are 3D DICOM files — this is a proof of concept.
- The clinical notes are matched to images by label (cancer=cancer note, normal=normal note) rather than being from the same patient. A real deployment would have actual patient-image-note triplets.
- 90% accuracy sounds great until you remember that in cancer diagnosis, the 10% that's wrong really matters. This is a research project, not a diagnostic device.
- We did not cure cancer. We want to be clear about that.

---

##  Acknowledgements

- **The one `/`** that caused 3 days of debugging — you taught us patience

---

## License

This project is for educational and research purposes.

**DeepOncoVision** · Multimodal Cancer Detection · CNN + BERT + Fusion · 2026
