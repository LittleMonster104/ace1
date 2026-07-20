#!/usr/bin/env python3
"""
Tent (Entropy Minimization) for Cross-Modal Retrieval

Adaptation strategy:
- Batch-wise adaptation by minimizing ranking entropy
- Loss: Row-wise entropy of similarity matrix (I2T direction)
- Update: Only batch normalization parameters
- Use standard test-time optimization

Reference: Wang et al. "Tent: Fully Test-Time Adaptation by Entropy Minimization" ICLR'21
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
import copy

device = "cpu"  # Change to "mps" for Mac M chip GPU acceleration

print("="*80)
print("Tent (Entropy Minimization) for Cross-Modal Retrieval")
print("="*80)
print(f"Device: {device}")

# ============================================================================
# Configuration
# ============================================================================
MODEL_NAME = "ViT-B/32"
TENT_LR = 1e-3
TENT_STEPS = 10
BATCH_SIZE = 32
TEMPERATURE = 0.01  # For entropy calculation

print(f"\nConfiguration:")
print(f"  Model: CLIP {MODEL_NAME}")
print(f"  Tent learning rate: {TENT_LR}")
print(f"  Tent steps: {TENT_STEPS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Temperature: {TEMPERATURE}")

# ============================================================================
# Load CLIP Model
# ============================================================================
print(f"\nLoading CLIP {MODEL_NAME}...")
model, preprocess = clip.load(MODEL_NAME, device=device, jit=False)
print("  ✅ CLIP model loaded")

# ============================================================================
# Tent Adaptation Functions
# ============================================================================
def collect_bn_params(model):
    """Collect batch normalization parameters for adaptation"""
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            for np, p in m.named_parameters():
                if p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names

def configure_model_for_tent(model):
    """Configure model for Tent adaptation"""
    model.eval()
    # Enable gradient only for BN parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Collect BN parameters
    params, names = collect_bn_params(model)
    
    # Enable gradient for BN parameters
    for param in params:
        param.requires_grad = True
    
    # Set BN layers to train mode to update running stats
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            m.train()
    
    print(f"  Tent: Found {len(params)} adaptable BN parameters")
    if len(params) == 0:
        print("  Warning: No BN parameters found. Using LayerNorm parameters instead.")
        # For CLIP ViT which uses LayerNorm, collect those parameters
        for nm, m in model.named_modules():
            if isinstance(m, nn.LayerNorm):
                for np, p in m.named_parameters():
                    p.requires_grad = True
                    params.append(p)
                    names.append(f"{nm}.{np}")
        print(f"  Tent: Found {len(params)} adaptable LayerNorm parameters")
    
    return params, names

def tent_adapt_batch(model, image_batch, text_features, optimizer, steps=10):
    """
    Adapt model on a batch by minimizing entropy
    
    Args:
        model: CLIP model
        image_batch: [B, 3, H, W] batch of images
        text_features: [M, D] all text features
        optimizer: optimizer for BN params
        steps: number of adaptation steps
    
    Returns:
        adapted_image_features: [B, D] adapted image features
    """
    for step in range(steps):
        optimizer.zero_grad()
        
        # Forward pass
        image_features = model.encode_image(image_batch)
        image_features = F.normalize(image_features, dim=-1)
        
        # Compute similarity matrix
        similarities = (image_features @ text_features.T) / TEMPERATURE  # [B, M]
        
        # Compute row-wise entropy (for each image, entropy over all texts)
        probs = F.softmax(similarities, dim=1)  # [B, M]
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)  # [B]
        
        # Loss is mean entropy
        loss = entropy.mean()
        
        # Backward and update
        loss.backward()
        optimizer.step()
    
    # Get final adapted features
    with torch.no_grad():
        image_features = model.encode_image(image_batch)
        image_features = F.normalize(image_features, dim=-1)
    
    return image_features

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

# Reset model to eval mode for baseline
model.eval()
for param in model.parameters():
    param.requires_grad = False

image_features_baseline = []
text_features_list = []
image_ids = []
image_paths = []

print("\nProcessing images...")
for img_data in tqdm(test_images, desc="Images"):
    try:
        filepath = img_data['filepath']
        filename = img_data['filename']
        img_path = DATA_DIR / filepath / filename
        
        if not img_path.exists():
            continue
        
        # Store path for later
        image_paths.append(img_path)
        
        # Load and preprocess image
        image = Image.open(img_path).convert('RGB')
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
print("Baseline Evaluation (No Tent)")
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
# Tent Evaluation (With Adaptation)
# ============================================================================
print("\n" + "="*80)
print("Tent Evaluation (With Entropy Minimization)")
print("="*80)

print("\nAdapting model with Tent...")
start_time = time.time()

# Create a copy of model for Tent adaptation
model_tent = copy.deepcopy(model)
params, names = configure_model_for_tent(model_tent)

if len(params) == 0:
    print("ERROR: No adaptable parameters found! Tent requires BN or LayerNorm layers.")
    print("Falling back to baseline results.")
    tent_r1 = baseline_r1
    tent_r5 = baseline_r5
    tent_r10 = baseline_r10
else:
    # Optimizer for BN parameters
    optimizer = torch.optim.Adam(params, lr=TENT_LR)
    
    # Process images in batches
    image_features_tent = []
    
    for batch_start in tqdm(range(0, n_images, BATCH_SIZE), desc="Tent Adaptation"):
        batch_end = min(batch_start + BATCH_SIZE, n_images)
        batch_paths = image_paths[batch_start:batch_end]
        
        # Load batch of images
        batch_images = []
        for img_path in batch_paths:
            image = Image.open(img_path).convert('RGB')
            image_input = preprocess(image)
            batch_images.append(image_input)
        
        batch_images = torch.stack(batch_images).to(device)  # [B, 3, H, W]
        
        # Adapt on this batch
        adapted_features = tent_adapt_batch(
            model_tent,
            batch_images,
            text_features.to(device),
            optimizer,
            steps=TENT_STEPS
        )
        
        image_features_tent.append(adapted_features.cpu())
    
    image_features_tent = torch.cat(image_features_tent, dim=0)  # [N_images, D]
    
    elapsed_time = time.time() - start_time
    print(f"\nTent adaptation time: {elapsed_time:.1f}s ({elapsed_time/n_images:.2f}s per image)")
    
    # Evaluate Tent results
    sims_tent = image_features_tent @ text_features.T  # [N_images, N_texts]
    
    ranks_tent = []
    for i in range(n_images):
        correct_text_indices = (image_ids == i).nonzero(as_tuple=True)[0]
        sim_row = sims_tent[i]
        max_correct_sim = sim_row[correct_text_indices].max().item()
        rank = (sim_row >= max_correct_sim).sum().item() - 1
        ranks_tent.append(rank)
    
    ranks_tent = np.array(ranks_tent)
    tent_r1 = (ranks_tent == 0).mean() * 100
    tent_r5 = (ranks_tent < 5).mean() * 100
    tent_r10 = (ranks_tent < 10).mean() * 100
    
    print(f"\nTent I2T Results:")
    print(f"  R@1:  {tent_r1:.2f}%")
    print(f"  R@5:  {tent_r5:.2f}%")
    print(f"  R@10: {tent_r10:.2f}%")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("Summary")
print("="*80)

print(f"\nBaseline (No Tent):")
print(f"  I2T R@1: {baseline_r1:.2f}%")

print(f"\nTent (Entropy Minimization):")
print(f"  I2T R@1: {tent_r1:.2f}%")

print(f"\nGain:")
print(f"  Absolute: {tent_r1 - baseline_r1:+.2f}%")
if baseline_r1 > 0:
    print(f"  Relative: {((tent_r1 / baseline_r1) - 1) * 100:+.2f}%")

# ============================================================================
# Save Results
# ============================================================================
results = {
    "model": MODEL_NAME,
    "dataset": "COCO",
    "n_images": n_images,
    "n_texts": n_texts,
    "tent_lr": TENT_LR,
    "tent_steps": TENT_STEPS,
    "batch_size": BATCH_SIZE,
    "baseline": {
        "i2t_r1": float(baseline_r1),
        "i2t_r5": float(baseline_r5),
        "i2t_r10": float(baseline_r10)
    },
    "tent": {
        "i2t_r1": float(tent_r1),
        "i2t_r5": float(tent_r5),
        "i2t_r10": float(tent_r10)
    },
    "gain": {
        "absolute": float(tent_r1 - baseline_r1),
        "relative_pct": float(((tent_r1 / baseline_r1) - 1) * 100) if baseline_r1 > 0 else 0.0
    },
    "time": {
        "total_seconds": float(elapsed_time) if len(params) > 0 else 0.0,
        "per_image_seconds": float(elapsed_time / n_images) if len(params) > 0 else 0.0
    }
}

output_dir = Path("/Users/jiazhu/Documents/ZJNU/EvoScientist/workspace/finalcode/results")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "baseline_tent_vitb32.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved to: {output_file}")
print("\n" + "="*80)
print("Tent Evaluation Complete")
print("="*80)
