#!/usr/bin/env python3
"""
TPT (Test-Time Prompt Tuning) for Cross-Modal Retrieval

Adaptation strategy:
- For each query image, optimize learnable prompt tokens
- Loss: Minimize entropy of similarity distribution over all texts
- Update: Only prompt parameters (frozen encoder)
- Steps: 10 optimization steps per image

Reference: Shu et al. "Test-Time Prompt Tuning for Zero-Shot Generalization in Vision-Language Models" NeurIPS'22
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

device = "cpu"  # Change to "mps" for Mac M chip GPU acceleration

print("="*80)
print("TPT (Test-Time Prompt Tuning) for Cross-Modal Retrieval")
print("="*80)
print(f"Device: {device}")

# ============================================================================
# Configuration
# ============================================================================
MODEL_NAME = "ViT-B/32"
TPT_STEPS = 10
TPT_LR = 1e-3
TEMPERATURE = 0.01  # For entropy calculation
N_CTX = 4  # Number of learnable prompt tokens
BATCH_SIZE = 32

print(f"\nConfiguration:")
print(f"  Model: CLIP {MODEL_NAME}")
print(f"  TPT steps: {TPT_STEPS}")
print(f"  TPT learning rate: {TPT_LR}")
print(f"  Context tokens: {N_CTX}")
print(f"  Temperature: {TEMPERATURE}")

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
# Learnable Prompt Module
# ============================================================================
class LearnablePrompt(nn.Module):
    """Learnable prompt tokens for test-time adaptation"""
    def __init__(self, n_ctx, ctx_dim, clip_model):
        super().__init__()
        self.n_ctx = n_ctx
        
        # Initialize learnable context vectors
        ctx_vectors = torch.empty(n_ctx, ctx_dim)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)
        
        # Store clip model for encoding
        self.clip_model = clip_model
        self.ctx_dim = ctx_dim
        
    def forward(self, text_tokens):
        """
        Args:
            text_tokens: [B, L] tokenized text
        Returns:
            text_features: [B, D] normalized text features
        """
        # Get text features with learnable prompts
        # This is a simplified version - in full TPT, prompts are prepended to text embeddings
        prefix = self.ctx.unsqueeze(0).expand(text_tokens.size(0), -1, -1)  # [B, n_ctx, D]
        
        # Get original text embeddings
        with torch.no_grad():
            x = self.clip_model.token_embedding(text_tokens).type(self.clip_model.dtype)  # [B, L, D]
        
        # Prepend learnable context
        # Take only the part after prompt tokens
        x = torch.cat([prefix, x[:, self.n_ctx:, :]], dim=1)  # [B, L, D]
        
        # Pass through transformer
        x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)  # [L, B, D]
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # [B, L, D]
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)
        
        # Take features from the eot token
        text_features = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)] @ self.clip_model.text_projection
        
        # Normalize
        text_features = F.normalize(text_features, dim=-1)
        
        return text_features

# ============================================================================
# TPT Adaptation Function
# ============================================================================
def tpt_adapt_image(image_feature, text_features, text_tokens, clip_model):
    """
    Adapt prompt for a single image using entropy minimization
    
    Args:
        image_feature: [D] normalized image feature
        text_features: [M, D] all text features (for initialization)
        text_tokens: [M, L] tokenized texts
        clip_model: CLIP model
    
    Returns:
        adapted_similarities: [M] adapted similarity scores
    """
    # Initialize learnable prompt
    ctx_dim = clip_model.ln_final.weight.shape[0]
    prompt_learner = LearnablePrompt(N_CTX, ctx_dim, clip_model).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(prompt_learner.parameters(), lr=TPT_LR)
    
    # Adaptation loop
    prompt_learner.train()
    for step in range(TPT_STEPS):
        optimizer.zero_grad()
        
        # Forward pass with learnable prompts
        adapted_text_features = prompt_learner(text_tokens)  # [M, D]
        
        # Compute similarity
        similarities = (image_feature @ adapted_text_features.T) / TEMPERATURE  # [M]
        
        # Entropy loss (minimize entropy = increase confidence)
        probs = F.softmax(similarities, dim=0)
        entropy = -(probs * torch.log(probs + 1e-8)).sum()
        
        loss = entropy
        loss.backward()
        optimizer.step()
    
    # Get final adapted similarities
    prompt_learner.eval()
    with torch.no_grad():
        adapted_text_features = prompt_learner(text_tokens)
        adapted_similarities = image_feature @ adapted_text_features.T
    
    return adapted_similarities.cpu()

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
# Extract Features
# ============================================================================
print("\n" + "="*80)
print("Extracting Features")
print("="*80)

image_features_list = []
text_features_list = []
text_tokens_list = []
image_ids = []

print("\nProcessing images...")
for img_data in tqdm(test_images, desc="Images"):
    try:
        filepath = img_data['filepath']
        filename = img_data['filename']
        img_path = DATA_DIR / filepath / filename
        
        if not img_path.exists():
            continue
        
        # Load and preprocess image
        image = Image.open(img_path).convert('RGB')
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        # Extract image feature
        with torch.no_grad():
            image_feature = model.encode_image(image_input)
            image_feature = F.normalize(image_feature, dim=-1).squeeze(0)
        
        # Extract text features and tokens for this image
        captions = [sent['raw'] for sent in img_data['sentences']]
        text_inputs = clip.tokenize(captions, truncate=True).to(device)
        
        with torch.no_grad():
            text_features = model.encode_text(text_inputs)
            text_features = F.normalize(text_features, dim=-1)
        
        # Store
        image_features_list.append(image_feature.cpu())
        text_features_list.extend([text_features[i].cpu() for i in range(len(captions))])
        text_tokens_list.extend([text_inputs[i].cpu() for i in range(len(captions))])
        image_ids.extend([len(image_features_list)-1] * len(captions))
    
    except Exception as e:
        print(f"Error processing image: {e}")
        continue

# Convert to tensors
image_features = torch.stack(image_features_list)  # [N_images, D]
text_features = torch.stack(text_features_list)    # [N_texts, D]
text_tokens = torch.stack(text_tokens_list)        # [N_texts, L]
image_ids = torch.tensor(image_ids)                # [N_texts]

n_images = len(image_features)
n_texts = len(text_features)

print(f"\nData statistics:")
print(f"  Images: {n_images}")
print(f"  Texts: {n_texts}")
print(f"  Avg captions per image: {n_texts / n_images:.1f}")

# ============================================================================
# Baseline Evaluation (No Adaptation)
# ============================================================================
print("\n" + "="*80)
print("Baseline Evaluation (No TPT)")
print("="*80)

sims_baseline = image_features @ text_features.T  # [N_images, N_texts]

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
# TPT Evaluation (With Adaptation)
# ============================================================================
print("\n" + "="*80)
print("TPT Evaluation (With Test-Time Prompt Tuning)")
print("="*80)

print("\nAdapting prompts for each image...")
start_time = time.time()

sims_tpt = []
for i in tqdm(range(n_images), desc="TPT Adaptation"):
    image_feature = image_features[i].to(device)
    
    # Adapt prompt for this image
    adapted_sims = tpt_adapt_image(
        image_feature,
        text_features.to(device),
        text_tokens.to(device),
        model
    )
    
    sims_tpt.append(adapted_sims)

sims_tpt = torch.stack(sims_tpt)  # [N_images, N_texts]

elapsed_time = time.time() - start_time
print(f"\nTPT adaptation time: {elapsed_time:.1f}s ({elapsed_time/n_images:.2f}s per image)")

# Evaluate TPT results
ranks_tpt = []
for i in range(n_images):
    correct_text_indices = (image_ids == i).nonzero(as_tuple=True)[0]
    sim_row = sims_tpt[i]
    max_correct_sim = sim_row[correct_text_indices].max().item()
    rank = (sim_row >= max_correct_sim).sum().item() - 1
    ranks_tpt.append(rank)

ranks_tpt = np.array(ranks_tpt)
tpt_r1 = (ranks_tpt == 0).mean() * 100
tpt_r5 = (ranks_tpt < 5).mean() * 100
tpt_r10 = (ranks_tpt < 10).mean() * 100

print(f"\nTPT I2T Results:")
print(f"  R@1:  {tpt_r1:.2f}%")
print(f"  R@5:  {tpt_r5:.2f}%")
print(f"  R@10: {tpt_r10:.2f}%")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("Summary")
print("="*80)

print(f"\nBaseline (No TPT):")
print(f"  I2T R@1: {baseline_r1:.2f}%")

print(f"\nTPT (Test-Time Prompt Tuning):")
print(f"  I2T R@1: {tpt_r1:.2f}%")

print(f"\nGain:")
print(f"  Absolute: {tpt_r1 - baseline_r1:+.2f}%")
print(f"  Relative: {((tpt_r1 / baseline_r1) - 1) * 100:+.2f}%")

# ============================================================================
# Save Results
# ============================================================================
results = {
    "model": MODEL_NAME,
    "dataset": "COCO",
    "n_images": n_images,
    "n_texts": n_texts,
    "tpt_steps": TPT_STEPS,
    "tpt_lr": TPT_LR,
    "baseline": {
        "i2t_r1": float(baseline_r1),
        "i2t_r5": float(baseline_r5),
        "i2t_r10": float(baseline_r10)
    },
    "tpt": {
        "i2t_r1": float(tpt_r1),
        "i2t_r5": float(tpt_r5),
        "i2t_r10": float(tpt_r10)
    },
    "gain": {
        "absolute": float(tpt_r1 - baseline_r1),
        "relative_pct": float(((tpt_r1 / baseline_r1) - 1) * 100)
    },
    "time": {
        "total_seconds": float(elapsed_time),
        "per_image_seconds": float(elapsed_time / n_images)
    }
}

output_dir = Path("/Users/jiazhu/Documents/ZJNU/EvoScientist/workspace/finalcode/results")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "baseline_tpt_vitb32.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved to: {output_file}")
print("\n" + "="*80)
print("TPT Evaluation Complete")
print("="*80)
