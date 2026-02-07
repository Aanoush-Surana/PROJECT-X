# 🚦 IDD20K Panoptic Segmentation & Prediction Pipeline
### Detectron2 • Panoptic FPN • Indian Traffic AI

> End-to-end computer vision pipeline for **Panoptic Segmentation**, **Model Fine-Tuning**, and **Real-Time Prediction** on Indian road environments.

---

## 🔍 Overview

This project builds a **full training → inference workflow** using **Detectron2 PanopticFPN (ResNet-50)** to understand urban traffic scenes including:

- Vehicles
- Pedestrians & Riders
- Autorickshaws
- Road Surfaces
- Infrastructure & Nature
- Real-Time Video Stream

Designed for **dense, occluded, real-world Indian traffic scenarios**.

---

## ✨ Core Capabilities

| Module | Description |
|-------|------------|
| Dataset Validation | Checks integrity & structure |
| Class Mapping | Contiguous ID generation (CUDA-safe) |
| Mask Generation | Polygon → Semantic & Panoptic masks |
| Fine-Tuning | Transfer learning from COCO weights |
| Predictor | Image & Video inference |
| Visualization | Color-coded segment overlays |

---

## 🧠 Model Architecture

```
PanopticFPN (ResNet-50 Backbone)
├── ROI Heads (Things)
└── Semantic Segmentation Head (Stuff)
```

- Pretrained on **COCO Panoptic**
- Fine-tuned on **IDD20K**
- Optimized for **GPU training**

---

## 📁 Dataset Structure

```text
idd20kII/
├── leftImg8bit/
│   └── train/
│       ├── city1/
│       │   ├── image1_leftImg8bit.jpg
│       │   └── ...
│       └── city2/
│           └── ...
└── gtFine/
    └── train/
        ├── city1/
        │   ├── image1_gtFine_polygons.json
        │   └── ...
        └── city2/
            └── ...
```

---

## ⚙️ Environment Setup

```bash
!nvidia-smi
!pip install torch torchvision torchaudio
!pip install detectron2
```

---

## 🏋️ Training Flow

1. Validate Dataset
2. Discover Classes
3. Generate Masks
4. Register Dataset
5. Configure Detectron2
6. Fine-Tune Model
7. Save Weights

Outputs:
```
/output/
 ├── model_0000250.pth
 ├── model_0000500.pth
 └── model_final.pth
```

---

## 🔮 Prediction

```python
outputs = predictor(image)
panoptic_seg, segments_info = outputs["panoptic_seg"]
```

Supports:
- Single Image
- Batch Images
- Video Frames

---

## 🎨 Visualization

- Panoptic colored overlays
- Thing vs Stuff separation
- Tiny segment filtering
- Saved PNG outputs

---

## 🚧 Common Errors & Fixes

| Error | Cause | Fix |
|------|------|-----|
| CUDA Assert | Non-contiguous class IDs | Run class mapping |
| Visualizer Index Error | Metadata mismatch | Align class lists |
| No Weights Found | Training incomplete | Check `/output` |

---

## 🚀 Future Enhancements

- Object Tracking (SORT / ByteTrack)
- Multi-GPU Distributed Training
- Autonomous Navigation Integration

---

## 📜 Acknowledgements

- Facebook AI Research – Detectron2  
- IDD Dataset Creators  
- COCO Panoptic Benchmark  

---

### Built for understanding roads where lanes are suggestions and chaos is data.
