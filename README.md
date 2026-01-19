# TaylorKAN-ViT

**TaylorKAN-ViT: Parameter-Efficient Vision Transformers for Medical Image Classification**

**Author:** Kaniz Fatema  
**Affiliation:** Wilfrid Laurier University

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Overview

TaylorKAN-ViT is an **ultra-compact Vision Transformer** designed for medical image classification under resource-constrained settings. Unlike conventional ViTs that rely on MLP-heavy architectures, TaylorKAN-ViT adopts a **KAN-first design** where all nonlinear mappings are replaced by Taylor-series-approximated Kolmogorov-Arnold Network (KAN) modules.

**Key Achievement:** Only **88.9K parameters** and **4.9G FLOPs** while maintaining competitive performance across multiple medical imaging benchmarks.

---

## ✨ Key Features

- **Ultra-Lightweight:** 88.9K parameters (~99.3% reduction vs. MedKAFormer, ~99.28% vs. MedViTV2)
- **KAN-First Design:** Replaces all MLP layers with Taylor-KAN modules
- **Medical Imaging Focus:** Optimized for limited-data clinical scenarios
- **Dual-Scale Learning:** Captures both patch-level features and global context
- **Resource-Efficient:** 4.9G FLOPs for edge/mobile deployment

---

## 📊 Performance

Tested on four diverse medical imaging benchmarks:

| Dataset | Domain | Classes | Accuracy | Parameters |
|---------|--------|---------|----------|------------|
| PneumoniaMNIST | Chest X-Ray | 2 | 94.36% | 88.9K |
| CPNX-ray | Radiology | 3 | 95.90% | 88.9K |
| PAD-UFES-20 | Dermatology | 6 | 61.00% | 88.9K |
| Kvasir | Endoscopy | 8 | 70.50% | 88.9K |

---

## 🏗️ Architecture

```
Input Image (224×224×3)
    ↓
Patch Embedding (16×16 patches → 196 tokens)
    ↓
Positional Embedding + Dropout
    ↓
Transformer Encoder (L=2 layers)
    ├── Single-Head Self-Attention
    │   └── LayerNorm + Attention + Residual
    └── TaylorKAN MLP Block
        ├── Taylor Series (3 terms, up to 5th order)
        ├── KAN Layer 1: 32→64 (B-spline basis)
        ├── GELU Activation
        ├── KAN Layer 2: 64→32 (B-spline basis)
        └── Residual Connection
    ↓
Global Average Pooling + LayerNorm
    ↓
Classification Head (with Dropout)
```

### Core Components

1. **CompactTaylorSeries:** Learnable-scale truncated Taylor approximation
2. **UltraCompactKANLinear:** B-spline basis functions with KAN formulation
3. **Single-Head Attention:** Efficient global context modeling
4. **Compact Design:** Embedding dim=32, MLP size=64, only 2 transformer layers

---

## 🚀 Usage

### Basic Example

```python
import torch
from taylorkan_vit_compact import TaylorKANViT

# Create model for medical imaging
model = TaylorKANViT(
    img_size=224,
    in_channels=3,
    patch_size=16,
    num_classes=3,           # e.g., Normal/Pneumonia/COVID-19
    embedding_dim=32,
    num_transformer_layers=2,
    mlp_size=64,
    taylor_terms=3
)

# Forward pass
x = torch.randn(1, 3, 224, 224)
output = model(x)
print(f"Output shape: {output.shape}")  # [1, 3]
```

### Training Configuration

Based on the paper's experimental setup:

```python
import torch.optim as optim
from torch.nn import CrossEntropyLoss

# Optimizer setup
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

# Learning rate scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=160,
    eta_min=0
)

# Loss function
criterion = CrossEntropyLoss(label_smoothing=0.1)

# Training parameters
EPOCHS = 160
BATCH_SIZE = 64
WARMUP_EPOCHS = 10
PATIENCE = 30  # for early stopping
```

---

## 📦 Requirements

- Python >= 3.8
- PyTorch >= 2.0
- torchvision

### Installation

```bash
pip install torch torchvision
```

---

## 🔬 Technical Details

### Model Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Image Size | 224×224 | Input resolution |
| Patch Size | 16×16 | Patch dimensions |
| Embedding Dim | 32 | Token dimension |
| Transformer Layers | 2 | Encoder depth |
| MLP Size | 64 | Hidden dimension |
| Attention Heads | 1 | Single-head attention |
| Taylor Terms | 3 | Up to 5th order |
| Grid Size | 3 | B-spline grid |
| Spline Order | 2 | Quadratic B-splines |

### Dropout Rates
- Embedding/Attention: 0.1
- TaylorKAN MLP: 0.15
- Classification Head: 0.2

---

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{fatema2025taylorkanvit,
  title={TaylorKAN-ViT: Enabling Parameter-Efficient Vision Transformers for Medical Image Classification},
  author={Fatema, Kaniz and Mohammed, Emad A. and Sehra, Sukhjit Singh},
  booktitle={International Conference on Engineering in Medicine and Biology (EMBC)},
  year={2025},
  organization={IEEE}
}
```

---

## 🙏 Acknowledgments

- Based on [Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756) by Liu et al.
- Inspired by [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929) by Dosovitskiy et al.
- Evaluated on [MedMNIST](https://medmnist.com/) and other medical imaging benchmarks

---

## 📄 License

MIT License - See LICENSE file for details

---

## 📧 Contact

**Kaniz Fatema**  
Department of Physics and Computer Science  
Wilfrid Laurier University  
Email: fate2180@mylaurier.ca

**GitHub:** [https://github.com/kanizbrishty/TaylorKAN-ViT-](https://github.com/kanizbrishty/TaylorKAN-ViT-)

---

⭐ **If you find this work helpful, please consider giving it a star!**
