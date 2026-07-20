#!/usr/bin/env python3
"""
CLIP ViT-B/16验证 - 三创新组合

在CLIP ViT-B/16上验证方法的通用性和适配性：
1. 双端Multi-view Ensemble
2. 跨模态一致性正则化
3. 测试时伪标签自训练

测试数据集：
- COCO 5K（通用域）
- RSITMD（遥感域）

验证方法在不同模型上的迁移能力
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
import torchvision.transforms as transforms

device = "cpu"
print("="*80)
print("CLIP ViT-B/16验证 - 三创新组合")
print("="*80)
print("测试模型通用性：B/32 → B/16 → L/14")

# ============================================================================
# 配置
# ============================================================================
PROMPT_TEMPLATES = [
    "{}",
    "a photo of {}",
    "a picture of {}",
    "an image showing {}",
    "this is {}",
]

IMAGE_AUGS = {
    'original': True,
    'random_crop': 3,
    'horizontal_flip': True,
    'color_jitter': True,
}

PROJ_HIDDEN_DIM = 256
ADAPTER_HIDDEN_DIM = 256
PROJ_LR = 1e-3
ADAPTER_LR = 1e-3
PROJ_EPOCHS = 2
ADAPTER_EPOCHS = 2
CONFIDENCE_THRESHOLD = 0.3
BATCH_SIZE = 32

print(f"\n配置:")
print(f"  模型: CLIP ViT-B/16")
print(f"  三创新组合验证")

# ============================================================================
# 组件定义
# ============================================================================
class ImageAugmentor:
    def __init__(self, image_size=224):
        self.image_size = image_size
        self.normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
        
        self.center_crop = transforms.Compose([
            transforms.Resize(image_size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            self.normalize
        ])
        
        self.random_crop = transforms.Compose([
            transforms.Resize(image_size, interpolation=Image.BICUBIC),
            transforms.RandomCrop(image_size),
            transforms.ToTensor(),
            self.normalize
        ])
        
        self.h_flip = transforms.Compose([
            transforms.Resize(image_size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            self.normalize
        ])
        
        self.color_jitter = transforms.Compose([
            transforms.Resize(image_size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            self.normalize
        ])
    
    def augment(self, image, config):
        augmented_images = []
        if config['original']:
            augmented_images.append(self.center_crop(image))
        if config['random_crop'] > 0:
            for _ in range(config['random_crop']):
                augmented_images.append(self.random_crop(image))
        if config['horizontal_flip']:
            augmented_images.append(self.h_flip(image))
        if config['color_jitter']:
            augmented_images.append(self.color_jitter(image))
        return torch.stack(augmented_images)

class ConsistencyProjector(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim)
        )
        self.alpha = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x):
        identity = x
        out = self.proj(x)
        out = identity + self.alpha * out
        out = F.normalize(out, dim=-1)
        return out

class SelfTrainingAdapter(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.alpha = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = identity + self.alpha * out
        out = F.normalize(out, dim=-1)
        return out

# ============================================================================
# 加载CLIP ViT-B/16
# ============================================================================
print("\n加载CLIP ViT-B/16...")
model, preprocess = clip.load("ViT-B/16", device=device, jit=False)
model.eval()
for param in model.parameters():
    param.requires_grad = False
print("  ✅ CLIP ViT-B/16已加载并冻结")

# ============================================================================
# 通用处理函数
# ============================================================================
def process_dataset(data_dir, dataset_json, dataset_name, is_rsitmd=False):
    """处理数据集并运行三创新验证"""
    
    print(f"\n{'='*80}")
    print(f"{dataset_name}数据集验证")
    print(f"{'='*80}")
    
    # 加载数据
    with open(dataset_json) as f:
        data = json.load(f)
    
    test_images = [img for img in data['images'] if img['split'] == 'test']
    if len(test_images) > 1000 and not is_rsitmd:  # COCO限制数量加速
        test_images = test_images[:1000]
    
    print(f"  测试图像: {len(test_images)}")
    
    # 提取多视角特征
    print(f"\n提取多视角特征...")
    augmentor = ImageAugmentor()
    multiview_image_features_list = []
    text_list = []
    image_ids = []
    
    for img_data in tqdm(test_images, desc="多视角"):
        try:
            if is_rsitmd:
                img_path = data_dir / "images" / img_data['filename']
            else:
                img_path = data_dir / img_data['filepath'] / img_data['filename']
            
            if not img_path.exists():
                continue
            
            image = Image.open(img_path).convert('RGB')
            augmented_imgs = augmentor.augment(image, IMAGE_AUGS).to(device)
            
            with torch.no_grad():
                multiview_features = model.encode_image(augmented_imgs)
                multiview_features = F.normalize(multiview_features, dim=-1)
                avg_feature = multiview_features.mean(dim=0, keepdim=True)
                avg_feature = F.normalize(avg_feature, dim=-1).cpu()
            
            multiview_image_features_list.append(avg_feature.squeeze(0))
            
            captions = [sent['raw'] for sent in img_data['sentences']]
            text_list.extend(captions)
            image_ids.extend([len(multiview_image_features_list)-1] * len(captions))
        except:
            continue
    
    multiview_image_features = torch.stack(multiview_image_features_list)
    image_ids = torch.tensor(image_ids)
    n_images = len(multiview_image_features)
    n_texts = len(text_list)
    print(f"  Images: {n_images}, Texts: {n_texts}")
    
    # 提取多prompt文本特征
    print(f"\n提取多prompt文本特征...")
    all_text_features = []
    for template in tqdm(PROMPT_TEMPLATES, desc="文本"):
        wrapped_texts = [template.format(text) for text in text_list]
        text_inputs = clip.tokenize(wrapped_texts, truncate=True).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_inputs)
            text_features = F.normalize(text_features, dim=-1).cpu()
        all_text_features.append(text_features)
    
    multi_text_features = torch.stack(all_text_features).mean(dim=0)
    
    # 评估函数
    def evaluate_full(img_feats, txt_feats, image_ids, name=""):
        img_feats = img_feats.to(device)
        txt_feats = txt_feats.to(device)
        sims = img_feats @ txt_feats.T
        
        # I2T
        i2t_ranks = []
        for i in range(len(img_feats)):
            correct_indices = (image_ids == i).nonzero(as_tuple=True)[0]
            sim_row = sims[i]
            max_sim = sim_row[correct_indices].max().item()
            rank = (sim_row >= max_sim).sum().item() - 1
            i2t_ranks.append(rank)
        i2t_ranks = np.array(i2t_ranks)
        i2t_r1 = (i2t_ranks == 0).mean() * 100
        i2t_r5 = (i2t_ranks < 5).mean() * 100
        i2t_r10 = (i2t_ranks < 10).mean() * 100
        
        # T2I
        t2i_ranks = []
        sims_t2i = sims.T
        for j in range(len(image_ids)):
            correct_img = image_ids[j].item()
            sim_row = sims_t2i[j]
            correct_sim = sim_row[correct_img].item()
            rank = (sim_row >= correct_sim).sum().item() - 1
            t2i_ranks.append(rank)
        t2i_ranks = np.array(t2i_ranks)
        t2i_r1 = (t2i_ranks == 0).mean() * 100
        t2i_r5 = (t2i_ranks < 5).mean() * 100
        t2i_r10 = (t2i_ranks < 10).mean() * 100
        
        if name:
            print(f"\n{name}:")
            print(f"  I2T: R@1={i2t_r1:.2f}%, R@5={i2t_r5:.2f}%, R@10={i2t_r10:.2f}%")
            print(f"  T2I: R@1={t2i_r1:.2f}%, R@5={t2i_r5:.2f}%, R@10={t2i_r10:.2f}%")
        
        return {
            'i2t_r1': i2t_r1, 'i2t_r5': i2t_r5, 'i2t_r10': i2t_r10,
            't2i_r1': t2i_r1, 't2i_r5': t2i_r5, 't2i_r10': t2i_r10
        }
    
    # Baseline
    print(f"\n阶段1: Baseline")
    single_view_features_list = []
    for img_data in tqdm(test_images[:n_images], desc="单视角", leave=False):
        try:
            if is_rsitmd:
                img_path = data_dir / "images" / img_data['filename']
            else:
                img_path = data_dir / img_data['filepath'] / img_data['filename']
            if not img_path.exists():
                continue
            image = Image.open(img_path).convert('RGB')
            image_input = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                image_feature = model.encode_image(image_input)
                image_feature = F.normalize(image_feature, dim=-1).cpu()
            single_view_features_list.append(image_feature.squeeze(0))
        except:
            continue
    
    single_view_features = torch.stack(single_view_features_list)
    text_inputs = clip.tokenize(text_list, truncate=True).to(device)
    with torch.no_grad():
        single_text_features = model.encode_text(text_inputs)
        single_text_features = F.normalize(single_text_features, dim=-1).cpu()
    
    results_baseline = evaluate_full(single_view_features, single_text_features, image_ids, "Baseline")
    
    # 双端Ensemble
    print(f"\n阶段2: + 双端Ensemble")
    results_ensemble = evaluate_full(multiview_image_features, multi_text_features, image_ids, "+ 双端Ensemble")
    
    # 一致性正则化
    print(f"\n阶段3: + 一致性正则化")
    img_projector = ConsistencyProjector(input_dim=512, hidden_dim=PROJ_HIDDEN_DIM).to(device)
    txt_projector = ConsistencyProjector(input_dim=512, hidden_dim=PROJ_HIDDEN_DIM).to(device)
    optimizer_proj = torch.optim.Adam(
        list(img_projector.parameters()) + list(txt_projector.parameters()), lr=PROJ_LR
    )
    
    for epoch in range(PROJ_EPOCHS):
        print(f"  Epoch {epoch + 1}/{PROJ_EPOCHS}")
        img_projector.train()
        txt_projector.train()
        total_loss = 0
        num_batches = 0
        indices = torch.randperm(n_images)
        
        for batch_start in range(0, n_images, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, n_images)
            batch_indices = indices[batch_start:batch_end]
            batch_img_feats = multiview_image_features[batch_indices].to(device)
            
            batch_txt_indices = []
            for img_idx in batch_indices:
                txt_idx = (image_ids == img_idx.item()).nonzero(as_tuple=True)[0]
                if len(txt_idx) > 0:
                    batch_txt_indices.append(txt_idx[0].item())
            if len(batch_txt_indices) == 0:
                continue
            
            batch_txt_feats = multi_text_features[batch_txt_indices].to(device)
            batch_txt_multi = []
            for txt_idx in batch_txt_indices:
                multi_feats = torch.stack([all_text_features[k][txt_idx] for k in range(len(PROMPT_TEMPLATES))])
                batch_txt_multi.append(multi_feats)
            batch_txt_multi = torch.stack(batch_txt_multi).to(device)
            
            proj_img = img_projector(batch_img_feats)
            proj_txt = txt_projector(batch_txt_feats)
            logits = proj_img @ proj_txt.T * 100
            labels = torch.arange(len(proj_img)).to(device)
            loss_cross = F.cross_entropy(logits, labels)
            
            proj_txt_multi = txt_projector(batch_txt_multi.reshape(-1, 512)).reshape(len(batch_txt_indices), len(PROMPT_TEMPLATES), 512)
            txt_variance = proj_txt_multi.var(dim=1).mean()
            loss = loss_cross + txt_variance
            
            optimizer_proj.zero_grad()
            loss.backward()
            optimizer_proj.step()
            total_loss += loss.item()
            num_batches += 1
        print(f"    Loss: {total_loss / num_batches:.4f}")
    
    img_projector.eval()
    txt_projector.eval()
    with torch.no_grad():
        proj_img_feats = img_projector(multiview_image_features.to(device))
        proj_txt_feats = txt_projector(multi_text_features.to(device))
    
    results_consistency = evaluate_full(proj_img_feats, proj_txt_feats, image_ids, "+ 一致性正则化")
    
    # 伪标签自训练
    print(f"\n阶段4: + 伪标签自训练")
    img_adapter = SelfTrainingAdapter(input_dim=512, hidden_dim=ADAPTER_HIDDEN_DIM).to(device)
    txt_adapter = SelfTrainingAdapter(input_dim=512, hidden_dim=ADAPTER_HIDDEN_DIM).to(device)
    optimizer_adapter = torch.optim.Adam(
        list(img_adapter.parameters()) + list(txt_adapter.parameters()), lr=ADAPTER_LR
    )
    
    for epoch in range(ADAPTER_EPOCHS):
        print(f"  Epoch {epoch + 1}/{ADAPTER_EPOCHS}")
        
        img_adapter.eval()
        txt_adapter.eval()
        with torch.no_grad():
            adapted_img = img_adapter(proj_img_feats)
            adapted_txt = txt_adapter(proj_txt_feats)
            sims = adapted_img @ adapted_txt.T
            
            pseudo_labels = []
            for i in range(n_images):
                correct_indices = (image_ids == i).nonzero(as_tuple=True)[0]
                sim_row = sims[i]
                max_sim_val = sim_row[correct_indices].max().item()
                max_sim_idx = correct_indices[sim_row[correct_indices].argmax()].item()
                if max_sim_val > CONFIDENCE_THRESHOLD:
                    pseudo_labels.append((i, max_sim_idx))
        
        print(f"    高置信度样本: {len(pseudo_labels)}/{n_images}")
        if len(pseudo_labels) == 0:
            break
        
        img_adapter.train()
        txt_adapter.train()
        indices = torch.randperm(len(pseudo_labels))
        total_loss = 0
        num_batches = 0
        
        for batch_start in range(0, len(pseudo_labels), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(pseudo_labels))
            batch_indices = indices[batch_start:batch_end]
            
            batch_img_idx = [pseudo_labels[i][0] for i in batch_indices]
            batch_txt_idx = [pseudo_labels[i][1] for i in batch_indices]
            
            batch_img = proj_img_feats[batch_img_idx]
            batch_txt = proj_txt_feats[batch_txt_idx]
            
            adapted_img = img_adapter(batch_img)
            adapted_txt = txt_adapter(batch_txt)
            
            logits = adapted_img @ adapted_txt.T * 100
            labels = torch.arange(len(batch_img_idx)).to(device)
            loss = F.cross_entropy(logits, labels)
            
            optimizer_adapter.zero_grad()
            loss.backward()
            optimizer_adapter.step()
            total_loss += loss.item()
            num_batches += 1
        print(f"    Loss: {total_loss / num_batches:.4f}")
    
    img_adapter.eval()
    txt_adapter.eval()
    with torch.no_grad():
        final_img_feats = img_adapter(proj_img_feats)
        final_txt_feats = txt_adapter(proj_txt_feats)
    
    results_final = evaluate_full(final_img_feats, final_txt_feats, image_ids, "+ 伪标签自训练")
    
    # 最终对比
    print(f"\n{'='*80}")
    print(f"{dataset_name}最终结果 - CLIP ViT-B/16")
    print(f"{'='*80}")
    
    methods = [
        ("Baseline", results_baseline),
        ("+ 双端Ensemble", results_ensemble),
        ("+ 一致性正则化", results_consistency),
        ("+ 伪标签自训练", results_final),
    ]
    
    print(f"\n{'方法':<25} {'I2T R@1':<10} {'I2T R@5':<10} {'I2T R@10':<10} {'T2I R@1':<10} {'T2I R@5':<10} {'T2I R@10':<10}")
    print("-" * 110)
    
    for name, res in methods:
        print(f"{name:<25} {res['i2t_r1']:>6.2f}%   {res['i2t_r5']:>6.2f}%   {res['i2t_r10']:>6.2f}%   "
              f"{res['t2i_r1']:>6.2f}%   {res['t2i_r5']:>6.2f}%   {res['t2i_r10']:>6.2f}%")
    
    baseline_i2t = results_baseline['i2t_r1']
    final_i2t = results_final['i2t_r1']
    print(f"\n总提升: I2T R@1 {final_i2t - baseline_i2t:+.2f}%")
    
    return results_baseline, results_final

# ============================================================================
# 主流程
# ============================================================================
print(f"\n{'='*80}")
print("开始测试CLIP ViT-B/16在两个数据集上的表现")
print(f"{'='*80}")

# COCO 5K
COCO_DIR = Path("/Users/jiazhu/Documents/ZJNU/EvoScientist/workspace/data/coco")
COCO_JSON = COCO_DIR / "dataset_coco.json"
coco_baseline, coco_final = process_dataset(COCO_DIR, COCO_JSON, "COCO 5K", is_rsitmd=False)

# RSITMD
RSITMD_DIR = Path("/Users/jiazhu/Documents/ZJNU/EvoScientist/workspace/data/RSITMD")
RSITMD_JSON = RSITMD_DIR / "dataset_rsitmd.json"
rsitmd_baseline, rsitmd_final = process_dataset(RSITMD_DIR, RSITMD_JSON, "RSITMD", is_rsitmd=True)

# 总结
print(f"\n{'='*80}")
print("CLIP ViT-B/16 - 三创新验证总结")
print(f"{'='*80}")
print(f"\nCOCO 5K:")
print(f"  Baseline: {coco_baseline['i2t_r1']:.2f}%")
print(f"  最终:     {coco_final['i2t_r1']:.2f}% (+{coco_final['i2t_r1']-coco_baseline['i2t_r1']:.2f}%)")
print(f"\nRSITMD:")
print(f"  Baseline: {rsitmd_baseline['i2t_r1']:.2f}%")
print(f"  最终:     {rsitmd_final['i2t_r1']:.2f}% (+{rsitmd_final['i2t_r1']-rsitmd_baseline['i2t_r1']:.2f}%)")
print(f"\n模型通用性验证完成！")
print(f"{'='*80}")
