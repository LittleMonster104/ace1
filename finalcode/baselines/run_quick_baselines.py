"""
Quick Baseline Test: Tent + MEMO on COCO 5K with ViT-B/32
Estimated time: 2-4 hours

Usage:
    python run_quick_baselines.py --output results/baselines_coco.json
"""

import torch
import torch.nn.functional as F
import clip
import numpy as np
from tqdm import tqdm
import json
import argparse
from pathlib import Path

# ==================== Dataset Loading ====================

def load_coco_5k(clip_preprocess, quick_test=False):
    """
    Load COCO 5K test set
    
    Args:
        quick_test: If True, only load 100 images for quick testing
    
    Returns:
        images: List of preprocessed images
        texts: List of text captions
        img_ids: List of image IDs (for I2T evaluation)
        txt_ids: List of text IDs (for T2I evaluation)
    """
    from pycocotools.coco import COCO
    from PIL import Image
    
    # Adjust paths to your setup
    coco_root = Path('./data/coco')
    ann_file = coco_root / 'annotations/captions_val2014.json'
    img_dir = coco_root / 'val2014'
    
    # Fallback paths if not found
    if not ann_file.exists():
        coco_root = Path('/Users/jiazhu/Documents/ZJNU/EvoScientist/workspace/data/coco')
        ann_file = coco_root / 'annotations/captions_val2014.json'
        img_dir = coco_root / 'val2014'
    
    if not ann_file.exists():
        raise FileNotFoundError(f"COCO annotations not found at {ann_file}")
    
    coco = COCO(ann_file)
    
    # Load test split
    img_ids = sorted(coco.imgs.keys())
    if quick_test:
        img_ids = img_ids[:100]  # Quick test: only 100 images
        print(f"Quick test mode: using {len(img_ids)} images")
    else:
        img_ids = img_ids[:5000]  # Full 5K test
    
    images = []
    texts = []
    img_to_txts = {}  # img_id -> list of text indices
    
    txt_idx = 0
    for img_id in tqdm(img_ids, desc='Loading COCO'):
        # Load image
        img_info = coco.imgs[img_id]
        img_path = img_dir / img_info['file_name']
        
        if not img_path.exists():
            print(f"Warning: Image {img_path} not found, skipping")
            continue
            
        image = Image.open(img_path).convert('RGB')
        image = clip_preprocess(image)
        images.append(image)
        
        # Load captions (5 per image)
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        img_to_txts[len(images)-1] = []
        for ann in anns:
            texts.append(ann['caption'])
            img_to_txts[len(images)-1].append(txt_idx)
            txt_idx += 1
    
    return torch.stack(images), texts, img_to_txts


# ==================== Tent Implementation ====================

class TentRetrieval:
    """
    Tent for retrieval: Minimize entropy of similarity distributions
    """
    def __init__(self, model, lr=1e-3, steps=1):
        self.model = model
        self.steps = steps
        
        # Collect trainable parameters (LayerNorm only)
        self.params = []
        for nm, m in model.visual.named_modules():
            if isinstance(m, torch.nn.LayerNorm):
                for p in m.parameters():
                    p.requires_grad = True
                    self.params.append(p)
        
        for nm, m in model.transformer.named_modules():
            if isinstance(m, torch.nn.LayerNorm):
                for p in m.parameters():
                    p.requires_grad = True
                    self.params.append(p)
        
        self.optimizer = torch.optim.Adam(self.params, lr=lr)
        print(f"Tent: Adapting {len(self.params)} LayerNorm parameters")
    
    def adapt_batch(self, images, text_features):
        """
        Adapt on one batch by minimizing entropy
        """
        for _ in range(self.steps):
            self.optimizer.zero_grad()
            
            # Encode images
            image_features = self.model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)
            
            # Compute similarities
            logits = image_features @ text_features.T / 0.07  # [B, M]
            
            # Entropy loss (minimize entropy = maximize confidence)
            probs = F.softmax(logits, dim=1)
            entropy = -(probs * torch.log(probs + 1e-5)).sum(dim=1).mean()
            
            loss = entropy
            loss.backward()
            self.optimizer.step()
    
    def extract_features(self, images, texts, batch_size=128):
        """
        Extract adapted features
        """
        device = next(self.model.parameters()).device
        
        # First encode all texts (frozen)
        print("Encoding texts...")
        text_features = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size)):
                batch_texts = texts[i:i+batch_size]
                text_tokens = clip.tokenize(batch_texts, truncate=True).to(device)
                feats = self.model.encode_text(text_tokens)
                feats = F.normalize(feats, dim=-1)
                text_features.append(feats.cpu())
        text_features = torch.cat(text_features, dim=0).to(device)
        
        # Adapt on images batch by batch
        print("Adapting on images with Tent...")
        image_features = []
        for i in tqdm(range(0, len(images), batch_size)):
            batch_imgs = images[i:i+batch_size].to(device)
            
            # Adapt
            self.model.train()
            self.adapt_batch(batch_imgs, text_features)
            
            # Extract features
            self.model.eval()
            with torch.no_grad():
                feats = self.model.encode_image(batch_imgs)
                feats = F.normalize(feats, dim=-1)
                image_features.append(feats.cpu())
        
        image_features = torch.cat(image_features, dim=0)
        text_features = text_features.cpu()
        
        return image_features, text_features


# ==================== MEMO Implementation ====================

class MEMORetrieval:
    """
    MEMO for retrieval: Marginal entropy minimization over augmentations
    Simplified: 8 augmentations instead of 64
    """
    def __init__(self, model, lr=1e-3, steps=1, n_aug=8):
        self.model = model
        self.steps = steps
        self.n_aug = n_aug
        
        # Same params as Tent
        self.params = []
        for nm, m in model.visual.named_modules():
            if isinstance(m, torch.nn.LayerNorm):
                for p in m.parameters():
                    p.requires_grad = True
                    self.params.append(p)
        
        for nm, m in model.transformer.named_modules():
            if isinstance(m, torch.nn.LayerNorm):
                for p in m.parameters():
                    p.requires_grad = True
                    self.params.append(p)
        
        self.optimizer = torch.optim.Adam(self.params, lr=lr)
        print(f"MEMO: Adapting {len(self.params)} parameters with {n_aug} augmentations")
        
        # Augmentation
        import torchvision.transforms as T
        self.aug_transform = T.Compose([
            T.RandomResizedCrop(224, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.3, 0.3, 0.3, 0.1),
            T.Normalize((0.48145466, 0.4578275, 0.40821073),
                       (0.26862954, 0.26130258, 0.27577711))
        ])
    
    def augment(self, images):
        """
        Generate n_aug augmented views
        """
        # images: [B, 3, 224, 224] already normalized
        # Need to denormalize first
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(images.device)
        
        images_denorm = images * std + mean
        
        aug_images = []
        for _ in range(self.n_aug):
            aug = self.aug_transform(images_denorm)
            aug_images.append(aug)
        
        return torch.stack(aug_images, dim=1)  # [B, n_aug, 3, 224, 224]
    
    def adapt_batch(self, images, text_features):
        """
        Adapt via marginal entropy minimization
        """
        # Generate augmentations
        aug_images = self.augment(images)  # [B, n_aug, 3, 224, 224]
        B, n_aug = aug_images.shape[:2]
        
        for _ in range(self.steps):
            self.optimizer.zero_grad()
            
            # Encode all augmented views
            aug_flat = aug_images.reshape(B * n_aug, 3, 224, 224)  # Use reshape instead of view
            aug_features = self.model.encode_image(aug_flat)
            aug_features = F.normalize(aug_features, dim=-1)
            aug_features = aug_features.view(B, n_aug, -1)  # [B, n_aug, D]
            
            # Compute similarities for each aug view
            logits_list = []
            for i in range(n_aug):
                logits = aug_features[:, i] @ text_features.T / 0.07  # [B, M]
                logits_list.append(logits)
            logits_all = torch.stack(logits_list, dim=1)  # [B, n_aug, M]
            
            # Marginal entropy: average prob over augs, then entropy
            avg_probs = F.softmax(logits_all, dim=2).mean(dim=1)  # [B, M]
            marginal_entropy = -(avg_probs * torch.log(avg_probs + 1e-5)).sum(dim=1).mean()
            
            loss = marginal_entropy
            loss.backward()
            self.optimizer.step()
    
    def extract_features(self, images, texts, batch_size=64):
        """
        Extract adapted features
        Note: Smaller batch size due to augmentations
        """
        device = next(self.model.parameters()).device
        
        # Encode texts
        print("Encoding texts...")
        text_features = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size*2)):
                batch_texts = texts[i:i+batch_size*2]
                text_tokens = clip.tokenize(batch_texts, truncate=True).to(device)
                feats = self.model.encode_text(text_tokens)
                feats = F.normalize(feats, dim=-1)
                text_features.append(feats.cpu())
        text_features = torch.cat(text_features, dim=0).to(device)
        
        # Adapt on images
        print("Adapting on images with MEMO...")
        image_features = []
        for i in tqdm(range(0, len(images), batch_size)):
            batch_imgs = images[i:i+batch_size].to(device)
            
            # Adapt
            self.model.train()
            self.adapt_batch(batch_imgs, text_features)
            
            # Extract features (no augmentation)
            self.model.eval()
            with torch.no_grad():
                feats = self.model.encode_image(batch_imgs)
                feats = F.normalize(feats, dim=-1)
                image_features.append(feats.cpu())
        
        image_features = torch.cat(image_features, dim=0)
        text_features = text_features.cpu()
        
        return image_features, text_features


# ==================== Evaluation ====================

def compute_recall(image_features, text_features, img_to_txts, k_vals=[1, 5, 10]):
    """
    Compute I2T and T2I recall@k
    """
    num_images = len(image_features)
    num_texts = len(text_features)
    
    # I2T: For each image, rank all texts
    print("Computing I2T recall...")
    i2t_ranks = []
    for i in tqdm(range(num_images)):
        # Compute similarities
        sims = image_features[i] @ text_features.T  # [num_texts]
        
        # Rank
        sorted_indices = torch.argsort(sims, descending=True)
        
        # Find rank of ground truth texts
        gt_txts = img_to_txts[i]
        ranks = []
        for gt_txt in gt_txts:
            rank = (sorted_indices == gt_txt).nonzero(as_tuple=True)[0].item()
            ranks.append(rank + 1)  # 1-indexed
        
        i2t_ranks.append(min(ranks))  # Best rank among GTs
    
    i2t_ranks = np.array(i2t_ranks)
    i2t_recalls = {k: (i2t_ranks <= k).mean() * 100 for k in k_vals}
    
    # T2I: For each text, rank all images
    print("Computing T2I recall...")
    txt_to_img = {}
    for img_idx, txt_indices in img_to_txts.items():
        for txt_idx in txt_indices:
            txt_to_img[txt_idx] = img_idx
    
    t2i_ranks = []
    for j in tqdm(range(num_texts)):
        sims = text_features[j] @ image_features.T  # [num_images]
        sorted_indices = torch.argsort(sims, descending=True)
        
        gt_img = txt_to_img[j]
        rank = (sorted_indices == gt_img).nonzero(as_tuple=True)[0].item()
        t2i_ranks.append(rank + 1)
    
    t2i_ranks = np.array(t2i_ranks)
    t2i_recalls = {k: (t2i_ranks <= k).mean() * 100 for k in k_vals}
    
    return i2t_recalls, t2i_recalls


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='baseline_results.json')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--quick-test', action='store_true', 
                       help='Quick test on 100 images only')
    args = parser.parse_args()
    
    if args.quick_test:
        print("="*60)
        print("QUICK TEST: Tent + MEMO on 100 COCO images (ViT-B/32)")
        print("="*60)
    else:
        print("="*60)
        print("Full Test: Tent + MEMO on COCO 5K (ViT-B/32)")
        print("="*60)
    
    device = args.device
    
    # Load model
    print("\n[1/4] Loading CLIP ViT-B/32...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Load data
    print("\n[2/4] Loading COCO...")
    images, texts, img_to_txts = load_coco_5k(preprocess, quick_test=args.quick_test)
    print(f"Loaded {len(images)} images, {len(texts)} texts")
    
    results = {}
    
    # Baseline (frozen)
    print("\n[3/4] Running Baseline (frozen)...")
    model.eval()
    with torch.no_grad():
        img_feats = model.encode_image(images.to(device))
        img_feats = F.normalize(img_feats, dim=-1).cpu()
        
        txt_tokens = clip.tokenize(texts, truncate=True).to(device)
        txt_feats = []
        for i in range(0, len(txt_tokens), 128):
            batch = txt_tokens[i:i+128]
            feats = model.encode_text(batch)
            txt_feats.append(F.normalize(feats, dim=-1).cpu())
        txt_feats = torch.cat(txt_feats, dim=0)
    
    i2t, t2i = compute_recall(img_feats, txt_feats, img_to_txts)
    results['Baseline'] = {'I2T': i2t, 'T2I': t2i}
    print(f"Baseline I2T R@1: {i2t[1]:.2f}%")
    
    # Tent
    print("\n[4/4a] Running Tent...")
    model_tent, _ = clip.load("ViT-B/32", device=device)
    tent = TentRetrieval(model_tent, lr=1e-3, steps=1)
    img_feats, txt_feats = tent.extract_features(images, texts)
    i2t, t2i = compute_recall(img_feats, txt_feats, img_to_txts)
    results['Tent'] = {'I2T': i2t, 'T2I': t2i}
    print(f"Tent I2T R@1: {i2t[1]:.2f}%")
    
    # MEMO
    print("\n[4/4b] Running MEMO...")
    model_memo, _ = clip.load("ViT-B/32", device=device)
    memo = MEMORetrieval(model_memo, lr=1e-3, steps=1, n_aug=8)
    img_feats, txt_feats = memo.extract_features(images, texts)
    i2t, t2i = compute_recall(img_feats, txt_feats, img_to_txts)
    results['MEMO'] = {'I2T': i2t, 'T2I': t2i}
    print(f"MEMO I2T R@1: {i2t[1]:.2f}%")
    
    # Save results
    print(f"\nSaving results to {args.output}")
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Method':<15} {'I2T R@1':<10} {'T2I R@1':<10}")
    print("-"*60)
    for method in ['Baseline', 'Tent', 'MEMO']:
        i2t_r1 = results[method]['I2T'][1]
        t2i_r1 = results[method]['T2I'][1]
        print(f"{method:<15} {i2t_r1:<10.2f} {t2i_r1:<10.2f}")
    print("="*60)
    
    return results


if __name__ == '__main__':
    main()

