# Image Caption Generator

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)
![CUDA](https://img.shields.io/badge/CUDA-Supported-76B900?style=for-the-badge&logo=nvidia&logoColor=white)

**A deep learning system that generates natural language descriptions of images using a CNN–Transformer architecture with cross-attention.**

[Overview](#-overview) · [Architecture](#-architecture) · [Getting Started](#-getting-started) · [Training](#-training) · [Results](#-results) · [Project Structure](#-project-structure)

</div>

---

## Overview

Image Caption Generator combines the feature-extraction power of a **ResNet50 CNN encoder** with the sequence modeling capabilities of a **Transformer decoder** featuring cross-attention. Given any image, the model generates a fluent, descriptive caption in natural language.

Trained and evaluated on the **Flickr8k** dataset, this project demonstrates an end-to-end deep learning pipeline — from vocabulary construction and data augmentation to mixed-precision training, checkpointing, and BLEU-score evaluation.

### Key Highlights

| Feature | Details |
|---|---|
| **Encoder** | ResNet50 pretrained on ImageNet (frozen backbone + learned projection) |
| **Decoder** | 6-layer Transformer with causal self-attention + cross-attention |
| **Training** | Mixed-precision (AMP), gradient accumulation, cosine LR scheduling |
| **Evaluation** | Corpus-level BLEU score via NLTK |
| **Generation** | Greedy decoding & Top-K sampling with temperature control |
| **Dataset** | Flickr8k (8,000 images, 5 captions each) |

---


### Model Configuration

| Hyperparameter | Default Value | Description |
|---|---|---|
| `n_embd` | 512 | Embedding dimension |
| `n_layer` | 6 | Number of Transformer blocks |
| `n_head` | 8 | Number of attention heads |
| `encoder_dim` | 2048 | CNN output dimension |
| `block_size` | 128 | Maximum caption length |
| `dropout` | 0.1 | Dropout rate |
| `vocab_size` | dynamic | Built from training captions |

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (recommended; CPU training is supported but slow)
- 8 GB+ RAM

### 1. Clone the Repository

```bash
git clone https://github.com/Boules123/Image-Caption-Transformers.git
cd Image-Caption-Transformers
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary>Full dependency list</summary>

```
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.0.0
numpy>=1.21.0
matplotlib>=3.5.0
tqdm>=4.65.0
nltk>=3.8.0
```

</details>

### 3. Download the Dataset

This project uses **Flickr8k**. Download it from [Kaggle – Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) and organize the files as follows:

```
data/
└── flickr8k/
    ├── Images/
    │   ├── 1000268201_693b08cb0e.jpg
    │   ├── 1001773457_577c3a7d70.jpg
    │   └── ...
    └── captions.txt
```

`captions.txt` format:

```
image,caption
1000268201_693b08cb0e.jpg,A child in a pink dress is climbing up a set of stairs in an entry way .
1000268201_693b08cb0e.jpg,A girl going into a wooden building .
...
```

### 4. Configure Paths

Edit `config.py` to point to your data directory:

```python
data_dir: str = "/data/flickr8k"
images_dir: str = "/data/flickr8k/Images"
captions_file: str = "/data/flickr8k/captions.txt"
```

---

## Training

### Quick Start

```bash
python train.py
```

### Custom Training Run

```bash
python train.py \
  --batch_size 64 \
  --epochs 15 \
  --lr 3e-4 \
  --data_dir /path/to/flickr8k
```

### Resume from Checkpoint

```bash
python train.py --resume checkpoints/checkpoint_epoch_10.pt
```

### Training Arguments

| Argument | Default | Description |
|---|---|---|
| `--batch_size` | 64 | Batch size for training |
| `--epochs` | 15 | Total number of training epochs |
| `--lr` | 3e-4 | Initial learning rate |
| `--resume` | None | Path to checkpoint to resume from |
| `--data_dir` | None | Override data directory |

### Training Pipeline

The training script executes the following steps automatically:

```
1. Build vocabulary from captions  (min_freq = 5)
       ↓
2. Split dataset  (80% train / 10% val / 10% test)
       ↓
3. Initialize model + AdamW optimizer + cosine LR scheduler
       ↓
4. Mixed-precision training loop  (AMP + GradScaler)
       ↓
5. Validate each epoch → save best_model.pt
       ↓
6. Save training history  (JSON)
       ↓
7. Evaluate on test set  (corpus BLEU)
```

### Data Augmentation

| Split | Augmentation |
|---|---|
| **Train** | Resize 256×256, Random Horizontal Flip, Normalize (ImageNet stats) |
| **Val / Test** | Resize 256×256, Normalize only |

---

## Results

### Training Curves

The model is evaluated on validation loss after each epoch. A typical training run looks like:

```
Epoch  1/15 | Train Loss: 3.6507 | Val Loss: 3.0470 | LR: 0.000300
Epoch  5/15 | Train Loss: 2.6328 | Val Loss: 2.6486 | LR: 0.000287
Epoch 10/15 | Train Loss: 1.8316 | Val Loss: 2.4211 | LR: 0.000245
Epoch 15/15 | Train Loss: 1.3141 | Val Loss: 2.4587 | LR: 0.000003
```

### BLEU Score

Evaluated at corpus level on the held-out test set using NLTK's `corpus_bleu` with Method 4 smoothing:

| Metric | Score |
|---|---|
| Corpus BLEU | ~0.21 – 0.26 |

> **Note:** BLEU scores vary with vocabulary size, training epochs, and batch size. The above range is a reference for 30-epoch runs on Flickr8k.

### Sample Predictions

| Image | Ground Truth | Generated Caption |
|---|---|---|
| `dog_beach.jpg` | *a brown dog is running on the sand near the water* | *a dog runs along the sandy beach* |
| `kids_park.jpg` | *two children are playing in a park on a sunny day* | *two young children play outside in the grass* |
| `man_bike.jpg` | *a man in a helmet is riding a bicycle on a path* | *a man rides a bike down a paved path* |

---

## Configuration Reference

All settings live in `config.py`. The most important ones:

### Model

```python
n_embd      = 512    # Transformer embedding dim
n_layer     = 6      # Decoder depth
n_head      = 8      # Attention heads (n_embd must be divisible by n_head)
encoder_dim = 2048   # CNN feature dim (ResNet50 output)
block_size  = 128     # Max caption tokens
dropout     = 0.1
```

### Training

```python
batch_size      = 32
learning_rate   = 3e-4
max_epochs      = 15
grad_clip       = 1.0
grad_accum_steps = 1   # Effective batch = batch_size × grad_accum_steps
lr_scheduler    = "cosine"   # "cosine" | "step"
weight_decay    = 0.01
```

### Generation

```python
max_gen_len = 20
temperature = 1.0
top_k       = 0     # 0 = greedy; >0 = top-k sampling
```

---

## Advanced Usage

### Unfreeze the CNN Backbone

By default, ResNet50 weights are frozen. To fine-tune the entire encoder after a warm-up period, call:

```python
model.encoder.unfreeze()
```

This is useful in later training epochs once the decoder has stabilized.

### Custom Generation

```python
from model.model import ImageCaptionModel
from config import get_config

config = get_config()
model = ImageCaptionModel(config)
model.load_model("checkpoints/best_model.pt")
model.eval()

# Generate with top-k sampling
captions = model.generate(
    images,
    max_len=20,
    top_k=5,
    temperature=0.8
)
```

### Visualize Predictions

```python
from utils.utils import visualize_predictions

visualize_predictions(
    images, ground_truth_captions, generated_captions,
    vocab=vocab,
    num_samples=4,
    save_path="outputs/predictions.png"
)
```

---

## Logging

Every training run creates a timestamped log file under `logs/`:

```
2026-01-15 10:32:01 [INFO] starting experiment: image_caption_transformer_experiment
2026-01-15 10:32:03 [INFO] Vocabulary size: 2984
2026-01-15 10:32:03 [INFO] train split: 30000 samples
2026-01-15 10:32:03 [INFO] val split: 3750 samples
2026-01-15 10:32:03 [INFO] test split: 3750 samples
2026-01-15 10:32:04 [INFO] num of parameters: 42,731,520
...
```

Training history (loss per epoch) is also saved as `logs/training_history.json` for offline analysis and plotting.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k) — Hodosh et al., 2013
- [Deep Visual-Semantic Alignments](https://cs.stanford.edu/people/karpathy/deepimagesent/) — Karpathy & Fei-Fei, 2015
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017
- PyTorch team for `scaled_dot_product_attention` and AMP support

---

<div align="center">
  Made with ❤️ by <strong>Boules Ashraf</strong> · MIT License · 2026
</div>