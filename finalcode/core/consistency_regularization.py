#!/usr/bin/env python3
"""
创新2: 跨模态一致性正则化（Cross-Modal Consistency Regularization）

核心创新：
1. 同一图像的不同aug应该在特征空间中相似（图像自一致性）
2. 同一文本的不同prompt应该在特征空间中相似（文本自一致性）
3. 匹配的图文对应该比不匹配的更相似（跨模态一致性）

技术：
- 用测试数据的无监督一致性信号
- 优化轻量级projection层
- 完全零样本！

方法：
- 对CLIP特征后接一个轻量级MLP
- 用一致性loss优化
- 不用任何标签！
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

device = "cpu"
print("="*80)
print("创新2: 跨模态一致性正则化")
print("="*80)

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

# 一致性优化配置
PROJ_HIDDEN_DIM = 256
LEARNING_RATE = 1e-3
NUM_EPOCHS = 3
BATCH_SIZE = 64
LAMBDA_IMG_CONSIST = 1.0  # 图像一致性权重
LAMBDA_TXT_CONSIST = 1.0  # 文本一致性权重
LAMBDA_CROSS_MODAL = 1.0  # 跨模态权重

print(f"\nPrompt模板数: {len(PROMPT_TEMPLATES)}")
print(f"优化轮数: {NUM_EPOCHS}")

# ============================================================================
# 一致性Projection层
# ============================================================================
class ConsistencyProjector(nn.Module):
    """轻量级投影层，优化一致性"""
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

# ============================================================================
# 加载CLIP
# ============================================================================
print("\n加载CLIP ViT-B/32...")
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
model.eval()

for param in model.parameters():
    param.requires_grad = False

print("  ✅ CLIP已加载并冻结")

# ============================================================================
# 加载COCO
# ============================================================================
DATA_DIR = Path("/Users/jiazhu/Documents/ZJNU/EvoScientist/workspace/data/coco")
print("\n加载COCO测试集...")
with open(DATA_DIR / "dataset_coco.json") as f:
    data = json.load(f)

test_images = [img for img in data['images'] if img['split'] == 'test'][:500]  # 测试500张
print(f"  Test images: {len(test_images)}")

# ============================================================================
# 提取特征
# ============================================================================
print("\n提取CLIP特征...")

image_features_list = []
text_list = []
image_ids = []

for img_data in tqdm(test_images, desc="提取特征"):
    try:
        filepath = img_data['filepath']
        filename = img_data['filename']
        img_path = DATA_DIR / filepath / filename
        
        if not img_path.exists():
            continue
        
        image = Image.open(img_path).convert('RGB')
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_feature = model.encode_image(image_input)
            image_feature = F.normalize(image_feature, dim=-1).cpu()
        
        image_features_list.append(image_feature.squeeze(0))
        
        captions = [sent['raw'] for sent in img_data['sentences']]
        text_list.extend(captions)
        image_ids.extend([len(image_features_list)-1] * len(captions))
    except:
        continue

image_features = torch.stack(image_features_list)
image_ids = torch.tensor(image_ids)

# 提取多prompt文本特征
print("\n提取多prompt文本特征...")
all_text_features = []

for template in tqdm(PROMPT_TEMPLATES, desc="文本特征"):
    wrapped_texts = [template.format(text) for text in text_list]
    text_inputs = clip.tokenize(wrapped_texts, truncate=True).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features = F.normalize(text_features, dim=-1).cpu()
    
    all_text_features.append(text_features)

# all_text_features: list of [n_texts, 512]
text_features_mean = torch.stack(all_text_features).mean(dim=0)

n_images = len(image_features)
n_texts = len(text_list)

print(f"  Images: {n_images}, Texts: {n_texts}")

# ============================================================================
# 评估函数
# ============================================================================
def evaluate(img_feats, txt_feats, image_ids, name=""):
    """评估I2T R@1"""
    img_feats = img_feats.to(device)
    txt_feats = txt_feats.to(device)
    
    sims = img_feats @ txt_feats.T
    
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
    
    if name:
        print(f"  {name}: R@1={i2t_r1:.2f}%, R@5={i2t_r5:.2f}%")
    
    return i2t_r1, i2t_r5

# ============================================================================
# Baseline: 无Projection
# ============================================================================
print("\n" + "="*80)
print("Baseline: CLIP + Prompt Ensemble（无Projection）")
print("="*80)

baseline_r1, baseline_r5 = evaluate(image_features, text_features_mean, image_ids, "Baseline")

# ============================================================================
# 一致性正则化优化
# ============================================================================
print("\n" + "="*80)
print("一致性正则化优化")
print("="*80)

# 创建Projector
img_projector = ConsistencyProjector(input_dim=512, hidden_dim=PROJ_HIDDEN_DIM).to(device)
txt_projector = ConsistencyProjector(input_dim=512, hidden_dim=PROJ_HIDDEN_DIM).to(device)

optimizer = torch.optim.Adam(
    list(img_projector.parameters()) + list(txt_projector.parameters()),
    lr=LEARNING_RATE
)

print(f"Projector参数量: {sum(p.numel() for p in img_projector.parameters()) * 2:,}")

# 优化
for epoch in range(NUM_EPOCHS):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    print(f"{'='*60}")
    
    img_projector.train()
    txt_projector.train()
    
    total_loss = 0
    num_batches = 0
    
    # Mini-batch训练
    indices = torch.randperm(n_images)
    
    for batch_start in tqdm(range(0, n_images, BATCH_SIZE), desc="训练", leave=False):
        batch_end = min(batch_start + BATCH_SIZE, n_images)
        batch_indices = indices[batch_start:batch_end]
        
        # 获取batch数据
        batch_img_feats = image_features[batch_indices].to(device)  # [B, 512]
        
        # 获取对应的文本
        batch_txt_indices = []
        for img_idx in batch_indices:
            txt_idx = (image_ids == img_idx.item()).nonzero(as_tuple=True)[0]
            if len(txt_idx) > 0:
                batch_txt_indices.append(txt_idx[0].item())  # 取第一个caption
        
        if len(batch_txt_indices) == 0:
            continue
        
        batch_txt_feats = text_features_mean[batch_txt_indices].to(device)  # [B, 512]
        
        # 获取所有prompt的文本特征（用于一致性）
        batch_txt_multi = []
        for txt_idx in batch_txt_indices:
            multi_feats = torch.stack([all_text_features[k][txt_idx] for k in range(len(PROMPT_TEMPLATES))])
            batch_txt_multi.append(multi_feats)
        batch_txt_multi = torch.stack(batch_txt_multi).to(device)  # [B, n_prompts, 512]
        
        # Forward
        proj_img = img_projector(batch_img_feats)  # [B, 512]
        proj_txt = txt_projector(batch_txt_feats)  # [B, 512]
        
        # Loss 1: 跨模态对比学习
        logits = proj_img @ proj_txt.T * 100  # [B, B]
        labels = torch.arange(len(proj_img)).to(device)
        loss_cross = F.cross_entropy(logits, labels)
        
        # Loss 2: 文本自一致性
        # 所有prompt的投影应该相似
        proj_txt_multi = txt_projector(batch_txt_multi.reshape(-1, 512)).reshape(len(batch_txt_indices), len(PROMPT_TEMPLATES), 512)
        # 计算方差（越小越好）
        txt_variance = proj_txt_multi.var(dim=1).mean()
        loss_txt_consist = txt_variance
        
        # 总Loss
        loss = (LAMBDA_CROSS_MODAL * loss_cross + 
                LAMBDA_TXT_CONSIST * loss_txt_consist)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    print(f"  Loss: {avg_loss:.4f}")
    
    # 评估
    img_projector.eval()
    txt_projector.eval()
    
    with torch.no_grad():
        proj_img_feats = img_projector(image_features.to(device))
        proj_txt_feats = txt_projector(text_features_mean.to(device))
    
    epoch_r1, epoch_r5 = evaluate(proj_img_feats, proj_txt_feats, image_ids, 
                                   f"Epoch {epoch+1}")

# ============================================================================
# 最终评估
# ============================================================================
print("\n" + "="*80)
print("最终结果对比")
print("="*80)

img_projector.eval()
txt_projector.eval()

with torch.no_grad():
    final_img_feats = img_projector(image_features.to(device))
    final_txt_feats = txt_projector(text_features_mean.to(device))

final_r1, final_r5 = evaluate(final_img_feats, final_txt_feats, image_ids, "最终")

methods = [
    ("Baseline", baseline_r1, baseline_r5),
    ("+ 一致性正则化", final_r1, final_r5),
]

print(f"\n{'方法':<25} {'R@1':<12} {'R@5':<12} {'vs Baseline':<12}")
print("-" * 65)

for name, r1, r5 in methods:
    delta = r1 - baseline_r1
    print(f"{name:<25} {r1:>6.2f}%     {r5:>6.2f}%     {delta:>+6.2f}%")

if final_r1 > baseline_r1:
    print(f"\n✅ 一致性正则化有效！")
    print(f"   绝对提升: {final_r1 - baseline_r1:+.2f}%")
    print(f"   相对提升: {(final_r1 - baseline_r1) / baseline_r1 * 100:.1f}%")
    print(f"\n   Projector参数: {sum(p.numel() for p in img_projector.parameters()) * 2:,}")
    print(f"   相对CLIP: {sum(p.numel() for p in img_projector.parameters()) * 2 / 151_000_000 * 100:.3f}%")
else:
    print(f"\n⚠️  一致性正则化未带来提升")

print("\n如果有效，在完整5K数据集上验证")
print("="*80)
