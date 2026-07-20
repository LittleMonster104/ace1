#!/usr/bin/env python3
"""
创新1: 图像端Multi-view Augmentation Ensemble

核心创新：
- 不仅在文本端用prompt ensemble
- 图像端也用多视角augmentation ensemble
- 双端ensemble，增强鲁棒性

技术细节：
- 对每张图像提取多个augmented视角
- 包括：原始、crop、flip、色彩变换
- 所有视角特征融合

完全零样本！不训练任何参数！
"""
import torch
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
print("创新1: 双端Multi-view Ensemble（图像+文本）")
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

# 图像augmentation配置
IMAGE_AUGS = {
    'original': True,      # 原始图像
    'center_crop': True,   # 中心裁剪
    'random_crop': 3,      # 3个随机裁剪
    'horizontal_flip': True, # 水平翻转
    'color_jitter': True,  # 色彩增强
}

print(f"\nPrompt模板数: {len(PROMPT_TEMPLATES)}")
print(f"图像augmentation数: {1 + 1 + 3 + 1 + 1} = 7")

# ============================================================================
# 图像Augmentation定义
# ============================================================================
class ImageAugmentor:
    """图像多视角augmentation"""
    def __init__(self, image_size=224):
        self.image_size = image_size
        
        # 基础变换
        self.normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
        
        # 裁剪变换
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
        
        # 翻转
        self.h_flip = transforms.Compose([
            transforms.Resize(image_size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            self.normalize
        ])
        
        # 色彩增强
        self.color_jitter = transforms.Compose([
            transforms.Resize(image_size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            self.normalize
        ])
    
    def augment(self, image, config):
        """生成多个augmented视角"""
        augmented_images = []
        
        # 1. 原始图像
        if config['original']:
            img = self.center_crop(image)
            augmented_images.append(img)
        
        # 2. 中心裁剪（已包含在原始中）
        
        # 3. 随机裁剪
        if config['random_crop'] > 0:
            for _ in range(config['random_crop']):
                img = self.random_crop(image)
                augmented_images.append(img)
        
        # 4. 水平翻转
        if config['horizontal_flip']:
            img = self.h_flip(image)
            augmented_images.append(img)
        
        # 5. 色彩增强
        if config['color_jitter']:
            img = self.color_jitter(image)
            augmented_images.append(img)
        
        return torch.stack(augmented_images)  # [n_augs, 3, 224, 224]

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

test_images = [img for img in data['images'] if img['split'] == 'test'][:500]  # 先测试500张
print(f"  Test images: {len(test_images)}")

# ============================================================================
# 提取多视角特征
# ============================================================================
print("\n提取多视角图像特征...")

augmentor = ImageAugmentor()

multiview_image_features_list = []
text_list = []
image_ids = []

for img_data in tqdm(test_images, desc="多视角特征"):
    try:
        filepath = img_data['filepath']
        filename = img_data['filename']
        img_path = DATA_DIR / filepath / filename
        
        if not img_path.exists():
            continue
        
        image = Image.open(img_path).convert('RGB')
        
        # 生成多个augmented视角
        augmented_imgs = augmentor.augment(image, IMAGE_AUGS).to(device)  # [n_augs, 3, 224, 224]
        
        with torch.no_grad():
            # 批量编码所有视角
            multiview_features = model.encode_image(augmented_imgs)  # [n_augs, 512]
            multiview_features = F.normalize(multiview_features, dim=-1)
            
            # 平均所有视角
            avg_feature = multiview_features.mean(dim=0, keepdim=True)  # [1, 512]
            avg_feature = F.normalize(avg_feature, dim=-1).cpu()
        
        multiview_image_features_list.append(avg_feature.squeeze(0))
        
        captions = [sent['raw'] for sent in img_data['sentences']]
        text_list.extend(captions)
        image_ids.extend([len(multiview_image_features_list)-1] * len(captions))
    
    except Exception as e:
        print(f"  错误: {e}")
        continue

multiview_image_features = torch.stack(multiview_image_features_list).to(device)
image_ids = torch.tensor(image_ids)
n_images = len(multiview_image_features)
n_texts = len(text_list)

print(f"  成功: {n_images}张图像, {n_texts}条文本")

# ============================================================================
# 评估函数
# ============================================================================
def evaluate(img_feats, txt_feats, image_ids, name=""):
    """评估I2T R@1和R@5"""
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
# Baseline: 单视角图像 + 单一文本
# ============================================================================
print("\n" + "="*80)
print("Baseline: 单视角图像 + 单一文本")
print("="*80)

# 重新提取单视角图像特征
print("提取单视角baseline特征...")
single_view_features_list = []

for img_data in tqdm(test_images[:len(multiview_image_features_list)], desc="单视角", leave=False):
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
        
        single_view_features_list.append(image_feature.squeeze(0))
    except:
        continue

single_view_features = torch.stack(single_view_features_list).to(device)

# 单一文本特征
text_inputs = clip.tokenize(text_list, truncate=True).to(device)
with torch.no_grad():
    single_text_features = model.encode_text(text_inputs)
    single_text_features = F.normalize(single_text_features, dim=-1).cpu()

baseline_r1, baseline_r5 = evaluate(single_view_features, single_text_features, image_ids,
                                    "Baseline")

# ============================================================================
# 方法2: 多视角图像 + 单一文本
# ============================================================================
print("\n" + "="*80)
print("方法2: 多视角图像 + 单一文本")
print("="*80)

multiview_r1, multiview_r5 = evaluate(multiview_image_features, single_text_features, image_ids,
                                      "+ 图像多视角")

# ============================================================================
# 方法3: 单视角图像 + 多prompt文本
# ============================================================================
print("\n" + "="*80)
print("方法3: 单视角图像 + 多prompt文本")
print("="*80)

# 多prompt文本特征
all_text_features = []
for template in tqdm(PROMPT_TEMPLATES, desc="Prompt", leave=False):
    wrapped_texts = [template.format(text) for text in text_list]
    text_inputs = clip.tokenize(wrapped_texts, truncate=True).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features = F.normalize(text_features, dim=-1).cpu()
    
    all_text_features.append(text_features)

multi_text_features = torch.stack(all_text_features).mean(dim=0)

prompt_r1, prompt_r5 = evaluate(single_view_features, multi_text_features, image_ids,
                                "+ 文本多prompt")

# ============================================================================
# 方法4: 多视角图像 + 多prompt文本（双端ensemble）
# ============================================================================
print("\n" + "="*80)
print("方法4: 多视角图像 + 多prompt文本（双端Ensemble）")
print("="*80)

dual_r1, dual_r5 = evaluate(multiview_image_features, multi_text_features, image_ids,
                            "+ 双端Ensemble")

# ============================================================================
# 最终对比
# ============================================================================
print("\n" + "="*80)
print("最终结果对比")
print("="*80)

methods = [
    ("Baseline (单视角+单文本)", baseline_r1, baseline_r5),
    ("+ 图像多视角", multiview_r1, multiview_r5),
    ("+ 文本多prompt", prompt_r1, prompt_r5),
    ("+ 双端Ensemble", dual_r1, dual_r5),
]

print(f"\n{'方法':<30} {'R@1':<12} {'R@5':<12} {'vs Baseline':<12}")
print("-" * 70)

for name, r1, r5 in methods:
    delta = r1 - baseline_r1
    print(f"{name:<30} {r1:>6.2f}%     {r5:>6.2f}%     {delta:>+6.2f}%")

best = max(methods, key=lambda x: x[1])
print(f"\n最佳方法: {best[0]}")
print(f"  R@1: {best[1]:.2f}% (提升{best[1]-baseline_r1:+.2f}%)")

if best[1] > baseline_r1:
    print(f"\n✅ 双端Ensemble有效！")
    print(f"   相对提升: {(best[1]-baseline_r1)/baseline_r1*100:.1f}%")
    
    # 分解贡献
    img_contrib = multiview_r1 - baseline_r1
    text_contrib = prompt_r1 - baseline_r1
    dual_contrib = dual_r1 - baseline_r1
    synergy = dual_contrib - (img_contrib + text_contrib)
    
    print(f"\n贡献分解:")
    print(f"  图像多视角贡献: +{img_contrib:.2f}%")
    print(f"  文本多prompt贡献: +{text_contrib:.2f}%")
    print(f"  协同效应: +{synergy:.2f}%")
    print(f"  总提升: +{dual_contrib:.2f}%")
else:
    print(f"\n⚠️  方法未带来提升")

print("\n如果有效，在完整5K数据集上验证")
print("="*80)
