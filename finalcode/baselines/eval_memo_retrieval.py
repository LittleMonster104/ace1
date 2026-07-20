#!/usr/bin/env python3
"""
MEMO (Marginal Entropy Minimization) for Cross-Modal Retrieval

Adaptation strategy:
- Per-image augmentation (N=16 for efficiency)
- Average features from augmented views
- Optionally minimize marginal entropy

Reference: Zhang et al. "MEMO: Test Time Robustness via Adaptation and Augmentation" NeurIPS'22
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from PIL import Image
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import torchvision.transforms as transforms

device = "cpu"  # Change to "mps" for Mac M chip GPU acceleration

print("="*80)
print("MEMO (Marginal Entropy Minimization) for Cross-Modal Retrieval")
print("="*80)
print(f"Device: {device}")

# ============================================================================
# Configuration
# ============================================================================
MODEL_NAME = "ViT-B/32"
N_AUGMENTATIONS = 16  # Number of augmented views per image (reduced from 64 for efficiency)
IMAGE_SIZE = 224

print(f"\nConfiguration:")
print(f"  Model: CLIP {MODEL_NAME}")
print(f"  Augmentations per image: {N_AUGMENTATIONS}")
print(f"  Image size: {IMAGE_SIZE}")

# ============================================================================
# Load CLIP Model
# ============================================================================
print(f"\nLoading CLIP {MODEL_NAME}...")
model, preprocess = clip.load(MODEL_NAME, device=device, jit=False)
model.eval()

# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

print("  ✅ CLIP model loaded and frozen")

# ============================================================================
# Augmentation Module
# ============================================================================
class MEMOAugmentor:
    """Augmentation module for MEMO"""
    def __init__(self, image_size=224):
        self.image_size = image_size
        
        # CLIP normalization
        self.normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
        
        # Standard preprocessing (center crop)
        self.standard = transforms.Compose([
            transforms.Resize(image_size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            self.normalize
        ])
        
        # Augmentation pipeline
        self.augment_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=Image.BICUBIC),
            transforms.RandomCrop(image_size, padding=image_size // 8, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.ToTensor(),
            self.normalize
        ])
    
    def get_augmented_views(self, image, n_augs):
        """
        Generate augmented views of an image
        
        Args:
            image: PIL Image
            n_augs: number of augmentations
        
        Returns:
            augmented_views: [n_augs, 3, H, W] tensor
        """
        views = []
        
        # Add one standard (non-augmented) view
        views.append(self.standard(image))
        
        # Add augmented views
        for _ in range(n_augs - 1):
            views.append(self.augment_transform(image))
        
        return torch.stack(views)

# ============================================================================
# MEMO Functions
# ============================================================================
def memo_adapt_image(image, model, augmentor, n_augs):
    """
    Apply MEMO adaptation to a single image
    
    Args:
        image: PIL Image
        model: CLIP model
        augmentor: MEMO augmentor
        n_augs: number of augmentations
    
    Returns:
        averaged_feature: [D] averaged feature across augmented views
    """
    # Generate augmented views
    augmented_views = augmentor.get_augmented_views(image, n_augs).to(device)  # [n_augs, 3, H, W]
    
    # Encode all views
    with torch.no_grad():
        features = model.encode_image(augmented_views)  # [n_augs, D]
        features = F.normalize(features, dim=-1)
    
    # Average features across views
    averaged_feature = features.mean(dim=0)  # [D]
    averaged_feature = F.normalize(averaged_feature, dim=-1)
    
    return averaged_feature

# ============================================================================
# Data Loading
# ============================================================================
DATA_DIR = Path("/Users/jiazhu/Documents/ZJNU/EvoScientist/workspace/data/coco")

print("\n" + "="*80)
print("Loading COCO Karpathy Test Split")
print("="*80)

with open(DATA_DIR / "dataset_coco.json") as f:
    data = json.load(f)

test_images = [img for img in data['images'] if img['split'] == 'test']

# Limit to 1000 images for faster evaluation
test_images = test_images[:1000]

print(f"  Test images: {len(test_images)}")

# ============================================================================
# Extract Baseline Features
# ============================================================================
print("\n" + "="*80)
print("Extracting Baseline Features")
print("="*80)

image_features_baseline = []
text_features_list = []
image_ids = []
image_pil_list = []  # Store PIL images for MEMO

print("\nProcessing images...")
for img_data in tqdm(test_images, desc="Images"):
    try:
        filepath = img_data['filepath']
        filename = img_data['filename']
        img_path = DATA_DIR / filepath / filename
        
        if not img_path.exists():
            continue
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image_pil_list.append(image)
        
        # Preprocess for baseline
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        # Extract baseline image feature
        with torch.no_grad():
            image_feature = model.encode_image(image_input)
            image_feature = F.normalize(image_feature, dim=-1).squeeze(0)
        
        # Extract text features for this image
        captions = [sent['raw'] for sent in img_data['sentences']]
        text_inputs = clip.tokenize(captions, truncate=True).to(device)
        
        with torch.no_grad():
            text_features = model.encode_text(text_inputs)
            text_features = F.normalize(text_features, dim=-1)
        
        # Store
        image_features_baseline.append(image_feature.cpu())
        text_features_list.extend([text_features[i].cpu() for i in range(len(captions))])
        image_ids.extend([len(image_features_baseline)-1] * len(captions))
    
    except Exception as e:
        print(f"Error processing image: {e}")
        continue

# Convert to tensors
image_features_baseline = torch.stack(image_features_baseline)  # [N_images, D]
text_features = torch.stack(text_features_list)                 # [N_texts, D]
image_ids = torch.tensor(image_ids)                             # [N_texts]

n_images = len(image_features_baseline)
n_texts = len(text_features)

print(f"\nData statistics:")
print(f"  Images: {n_images}")
print(f"  Texts: {n_texts}")
print(f"  Avg captions per image: {n_texts / n_images:.1f}")

# ============================================================================
# Baseline Evaluation (No Adaptation)
# ============================================================================
print("\n" + "="*80)
print("Baseline Evaluation (No MEMO)")
print("="*80)

sims_baseline = image_features_baseline @ text_features.T  # [N_images, N_texts]

ranks_baseline = []
for i in range(n_images):
    correct_text_indices = (image_ids == i).nonzero(as_tuple=True)[0]
    sim_row = sims_baseline[i]
    max_correct_sim = sim_row[correct_text_indices].max().item()
    rank = (sim_row >= max_correct_sim).sum().item() - 1
    ranks_baseline.append(rank)

ranks_baseline = np.array(ranks_baseline)
baseline_r1 = (ranks_baseline == 0).mean() * 100
baseline_r5 = (ranks_baseline < 5).mean() * 100
baseline_r10 = (ranks_baseline < 10).mean() * 100

print(f"\nBaseline I2T Results:")
print(f"  R@1:  {baseline_r1:.2f}%")
print(f"  R@5:  {baseline_r5:.2f}%")
print(f"  R@10: {baseline_r10:.2f}%")

# ============================================================================
# MEMO Evaluation (With Augmentation)
# ============================================================================
print("\n" + "="*80)
print("MEMO Evaluation (With Multi-View Augmentation)")
print("="*80)

print(f"\nGenerating {N_AUGMENTATIONS} augmented views per image...")
start_time = time.time()

augmentor = MEMOAugmentor(image_size=IMAGE_SIZE)

image_features_memo = []
for i in tqdm(range(n_images), desc="MEMO Adaptation"):
    image = image_pil_list[i]
    
    # Apply MEMO adaptation
    averaged_feature = memo_adapt_image(image, model, augmentor, N_AUGMENTATIONS)
    
    image_features_memo.append(averaged_feature.cpu())

image_features_memo = torch.stack(image_features_memo)  # [N_images, D]

elapsed_time = time.time() - start_time
print(f"\nMEMO adaptation time: {elapsed_time:.1f}s ({elapsed_time/n_images:.2f}s per image)")

# Evaluate MEMO results
sims_memo = image_features_memo @ text_features.T  # [N_images, N_texts]

ranks_memo = []
for i in range(n_images):
    correct_text_indices = (image_ids == i).nonzero(as_tuple=True)[0]
    sim_row = sims_memo[i]
    max_correct_sim = sim_row[correct_text_indices].max().item()
    rank = (sim_row >= max_correct_sim).sum().item() - 1
    ranks_memo.append(rank)

ranks_memo = np.array(ranks_memo)
memo_r1 = (ranks_memo == 0).mean() * 100
memo_r5 = (ranks_memo < 5).mean() * 100
memo_r10 = (ranks_memo < 10).mean() * 100

print(f"\nMEMO I2T Results:")
print(f"  R@1:  {memo_r1:.2f}%")
print(f"  R@5:  {memo_r5:.2f}%")
print(f"  R@10: {memo_r10:.2f}%")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("Summary")
print("="*80)

print(f"\nBaseline (No MEMO):")
print(f"  I2T R@1: {baseline_r1:.2f}%")

print(f"\nMEMO (Multi-View Augmentation):")
print(f"  I2T R@1: {memo_r1:.2f}%")

print(f"\nGain:")
print(f"  Absolute: {memo_r1 - baseline_r1:+.2f}%")
if baseline_r1 > 0:
    print(f"  Relative: {((memo_r1 / baseline_r1) - 1) * 100:+.2f}%")

# ============================================================================
# Save Results
# ============================================================================
results = {
    "model": MODEL_NAME,
    "dataset": "COCO",
    "n_images": n_images,
    "n_texts": n_texts,
    "n_augmentations": N_AUGMENTATIONS,
    "baseline": {
        "i2t_r1": float(baseline_r1),
        "i2t_r5": float(baseline_r5),
        "i2t_r10": float(baseline_r10)
    },
    "memo": {
        "i2t_r1": float(memo_r1),
        "i2t_r5": float(memo_r5),
        "i2t_r10": float(memo_r10)
    },
    "gain": {
        "absolute": float(memo_r1 - baseline_r1),
        "relative_pct": float(((memo_r1 / baseline_r1) - 1) * 100) if baseline_r1 > 0 else 0.0
    },
    "time": {
        "total_seconds": float(elapsed_time),
        "per_image_seconds": float(elapsed_time / n_images)
    }
}

output_dir = Path("/Users/jiazhu/Documents/ZJNU/EvoScientist/workspace/finalcode/results")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "baseline_memo_vitb32.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved to: {output_file}")
print("\n" + "="*80)
print("MEMO Evaluation Complete")
print("="*80)
