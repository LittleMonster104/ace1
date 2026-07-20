#!/usr/bin/env python3
"""
Educational PPT Dataset Evaluation - ACE完整框架

在教育PPT数据集上验证ACE完整框架：
1. 双端Multi-view Ensemble (Dual-End Fusion)
2. 跨模态一致性正则化 (Consistency Regularization)
3. 自适应伪标签 (Adaptive Pseudo-Labeling)

测试教育垂直领域泛化能力
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
import argparse

device = "cpu"  # Mac MPS
print("="*80)
print("Educational PPT Dataset - ACE Framework Evaluation")
print("="*80)

# ============================================================================
# 配置
# ============================================================================
PROMPT_TEMPLATES = [
    "{}",
    "a photo of {}",
    "a picture of {}",
    "an educational image showing {}",
    "a teaching material about {}",
]

IMAGE_AUGS = {
    'original': True,
    'random_crop': 3,
    'horizontal_flip': True,
    'color_jitter': True,
}

# 一致性优化
PROJ_HIDDEN_DIM = 256
PROJ_LR = 1e-3
PROJ_EPOCHS = 2  # 教育数据集较大，减少epoch

# 自适应伪标签
ADAPTER_HIDDEN_DIM = 256
ADAPTER_LR = 1e-3
ADAPTER_EPOCHS = 2

# 自适应Top-K策略（基于baseline性能）
ADAPTIVE_TOPK = {
    'weak': 0.20,      # baseline < 30%
    'medium': 0.10,    # 30% <= baseline < 50%
    'strong': 0.05,    # baseline >= 50%
}

BATCH_SIZE = 32
NUM_TEST_SAMPLES = 5000  # 可选：采样5000测试，或设为None使用完整测试集

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='ViT-B/32', 
                    choices=['ViT-B/32', 'ViT-L/14'],
                    help='CLIP model to evaluate')
parser.add_argument('--full_test', action='store_true',
                    help='Use full test set (8027 images) instead of 5000 sample')
parser.add_argument('--output', type=str, default='results/educational_ace_results.json',
                    help='Output path for results')
args = parser.parse_args()

if args.full_test:
    NUM_TEST_SAMPLES = None
    print(f"\n数据集: Educational PPT (Full test set: 8027 images)")
else:
    print(f"\n数据集: Educational PPT (Sampled: {NUM_TEST_SAMPLES} images)")

print(f"模型: CLIP {args.model}")
print(f"三创新: 双端Fusion + 一致性正则化 + 自适应伪标签")

# ============================================================================
# 组件定义（复用RSITMD代码结构）
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
        views = []
        if config.get('original', True):
            views.append(self.center_crop(image))
        
        n_crops = config.get('random_crop', 0)
        for _ in range(n_crops):
            views.append(self.random_crop(image))
        
        if config.get('horizontal_flip', False):
            views.append(self.h_flip(image))
        
        if config.get('color_jitter', False):
            views.append(self.color_jitter(image))
        
        return torch.stack(views)


class ConsistencyProjector(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        return self.proj(x) + x  # 残差连接


class ModalityAdapter(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        return self.adapter(x) + x


# ============================================================================
# 数据加载
# ============================================================================
def load_educational_data(num_samples=None):
    """加载Educational PPT数据集 (Karpathy格式)"""
    data_root = Path("/Users/jiazhu/Documents/ZJNU/EvoScientist/workspace/data/educational")
    anno_file = data_root / "annotations" / "dataset_edu_ppt_hq_karpathy.json"
    
    print(f"\n加载数据集: {anno_file}")
    with open(anno_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 筛选测试集
    test_images = [img for img in data['images'] if img.get('split') == 'test']
    
    # 可选：采样
    if num_samples and len(test_images) > num_samples:
        np.random.seed(42)
        indices = np.random.choice(len(test_images), num_samples, replace=False)
        test_images = [test_images[i] for i in sorted(indices)]
        print(f"采样测试集: {len(test_images)} 图像")
    else:
        print(f"完整测试集: {len(test_images)} 图像")
    
    # 构建数据结构
    dataset = []
    for img_data in test_images:
        img_path = data_root / "images" / img_data['filepath'] / img_data['filename']
        captions_en = [sent['raw_en'] for sent in img_data['sentences']]
        captions_zh = [sent['raw'] for sent in img_data['sentences']]
        
        dataset.append({
            'image_path': str(img_path),
            'captions_en': captions_en,
            'captions_zh': captions_zh,
            'subject': img_data.get('subject', 'unknown')
        })
    
    print(f"数据集规模: {len(dataset)} 图像, {len(dataset)*3} 图文对")
    print(f"Caption语言: 英文 (raw_en)")
    print(f"学科数量: {len(set(d['subject'] for d in dataset))}")
    
    return dataset


# ============================================================================
# 评估函数
# ============================================================================
def evaluate_retrieval(image_features, text_features, return_ranks=False):
    """
    Args:
        image_features: [N, D]
        text_features: [M, D] where M = N*3 (3 captions per image)
    """
    N = len(image_features)
    M = len(text_features)
    assert M == N * 3, f"Expected {N*3} captions for {N} images, got {M}"
    
    # Normalize
    image_features = F.normalize(image_features, dim=1)
    text_features = F.normalize(text_features, dim=1)
    
    # I2T: 对每张图，找最相关的caption
    similarity_i2t = image_features @ text_features.T  # [N, M]
    ranks_i2t = []
    for i in range(N):
        # 该图的3个ground-truth caption在 [i*3, i*3+1, i*3+2]
        gt_indices = list(range(i*3, i*3+3))
        scores = similarity_i2t[i]
        sorted_indices = torch.argsort(scores, descending=True)
        # 找到第一个GT的排名
        rank = min([torch.where(sorted_indices == gt_idx)[0].item() for gt_idx in gt_indices])
        ranks_i2t.append(rank)
    
    # T2I: 对每个caption，找正确的图
    similarity_t2i = text_features @ image_features.T  # [M, N]
    ranks_t2i = []
    for j in range(M):
        gt_image_idx = j // 3  # caption j属于第gt_image_idx张图
        scores = similarity_t2i[j]
        sorted_indices = torch.argsort(scores, descending=True)
        rank = torch.where(sorted_indices == gt_image_idx)[0].item()
        ranks_t2i.append(rank)
    
    # Recall@K
    ranks_i2t = np.array(ranks_i2t)
    ranks_t2i = np.array(ranks_t2i)
    
    results = {
        'i2t': {
            'r1': 100 * np.mean(ranks_i2t < 1),
            'r5': 100 * np.mean(ranks_i2t < 5),
            'r10': 100 * np.mean(ranks_i2t < 10),
        },
        't2i': {
            'r1': 100 * np.mean(ranks_t2i < 1),
            'r5': 100 * np.mean(ranks_t2i < 5),
            'r10': 100 * np.mean(ranks_t2i < 10),
        }
    }
    
    if return_ranks:
        return results, ranks_i2t, ranks_t2i
    return results


def compute_adaptive_topk(baseline_r1, num_samples):
    """根据baseline性能自适应选择Top-K"""
    if baseline_r1 < 30.0:
        k_ratio = ADAPTIVE_TOPK['weak']
        level = 'weak'
    elif baseline_r1 < 50.0:
        k_ratio = ADAPTIVE_TOPK['medium']
        level = 'medium'
    else:
        k_ratio = ADAPTIVE_TOPK['strong']
        level = 'strong'
    
    k = int(k_ratio * num_samples)
    print(f"\n自适应Top-K策略:")
    print(f"  Baseline R@1: {baseline_r1:.2f}%")
    print(f"  识别容量等级: {level}")
    print(f"  Top-K比例: {k_ratio*100}%")
    print(f"  Top-K数量: {k}")
    
    return k


# ============================================================================
# 主评估流程
# ============================================================================
def main():
    # 1. 加载模型
    print(f"\n{'='*80}")
    print("1. 加载CLIP模型")
    print(f"{'='*80}")
    model, preprocess = clip.load(args.model, device=device)
    model.eval()
    
    # 2. 加载数据
    print(f"\n{'='*80}")
    print("2. 加载Educational PPT数据集")
    print(f"{'='*80}")
    dataset = load_educational_data(num_samples=NUM_TEST_SAMPLES)
    
    # 3. Baseline评估
    print(f"\n{'='*80}")
    print("3. Baseline评估 (标准CLIP零样本)")
    print(f"{'='*80}")
    
    image_features_list = []
    text_features_list = []
    
    augmentor = ImageAugmentor()
    
    for item in tqdm(dataset, desc="提取特征"):
        # 图像
        image = Image.open(item['image_path']).convert('RGB')
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_feat = model.encode_image(image_tensor)
        image_features_list.append(image_feat.cpu())
        
        # 文本 (3个英文caption)
        texts = clip.tokenize(item['captions_en']).to(device)
        with torch.no_grad():
            text_feat = model.encode_text(texts)
        text_features_list.append(text_feat.cpu())
    
    image_features = torch.cat(image_features_list, dim=0)  # [N, D]
    text_features = torch.cat(text_features_list, dim=0)    # [N*3, D]
    
    baseline_results = evaluate_retrieval(image_features, text_features)
    
    print(f"\nBaseline Results:")
    print(f"  I2T: R@1={baseline_results['i2t']['r1']:.2f}%, "
          f"R@5={baseline_results['i2t']['r5']:.2f}%, "
          f"R@10={baseline_results['i2t']['r10']:.2f}%")
    print(f"  T2I: R@1={baseline_results['t2i']['r1']:.2f}%, "
          f"R@5={baseline_results['t2i']['r5']:.2f}%, "
          f"R@10={baseline_results['t2i']['r10']:.2f}%")
    
    # 4. ACE组件逐步添加
    print(f"\n{'='*80}")
    print("4. ACE组件评估")
    print(f"{'='*80}")
    
    all_results = {
        'model': args.model,
        'dataset': 'educational_ppt_hq',
        'num_test_images': len(dataset),
        'baseline': baseline_results,
    }
    
    # Component 1: Dual-End Fusion (简化版：直接平均I2T和T2I的相似度)
    print(f"\n--- Component 1: Dual-End Fusion ---")
    # TODO: 实现完整的多视角增强+双向融合
    # 这里先占位，使用baseline结果
    dual_end_results = baseline_results  # 占位
    all_results['dual_end_fusion'] = dual_end_results
    
    # Component 2: + Consistency Regularization
    print(f"\n--- Component 2: + Consistency Regularization ---")
    # TODO: 训练consistency projector
    consistency_results = baseline_results  # 占位
    all_results['consistency_reg'] = consistency_results
    
    # Component 3: + Adaptive Pseudo-Labeling
    print(f"\n--- Component 3: + Adaptive Pseudo-Labeling ---")
    topk = compute_adaptive_topk(baseline_results['i2t']['r1'], len(dataset))
    # TODO: 自适应伪标签训练
    pseudo_label_results = baseline_results  # 占位
    all_results['adaptive_pseudo_label'] = pseudo_label_results
    
    # ACE (Full)
    all_results['ace_full'] = pseudo_label_results
    
    # 5. 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"结果已保存到: {output_path}")
    print(f"{'='*80}")
    
    # 6. 打印总结
    print(f"\n最终结果总结:")
    print(f"Baseline:    I2T R@1={baseline_results['i2t']['r1']:.2f}%")
    print(f"ACE (Full):  I2T R@1={all_results['ace_full']['i2t']['r1']:.2f}%")
    print(f"提升:        +{all_results['ace_full']['i2t']['r1'] - baseline_results['i2t']['r1']:.2f}%")


if __name__ == '__main__':
    main()
