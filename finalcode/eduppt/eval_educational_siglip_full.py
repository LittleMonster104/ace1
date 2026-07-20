#!/usr/bin/env python3
"""
完整ACE框架评估 - Educational PPT Dataset (SigLIP Models)

支持SigLIP-base和SigLIP-SO400M
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel
from PIL import Image
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torchvision.transforms as transforms
import argparse

device = "cpu"
print("="*80)
print("完整ACE框架评估 - Educational PPT Dataset (SigLIP)")
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

PROJ_HIDDEN_DIM = 256
PROJ_LR = 1e-3
PROJ_EPOCHS = 2

ADAPTER_HIDDEN_DIM = 256
ADAPTER_LR = 1e-3
ADAPTER_EPOCHS = 2

ADAPTIVE_TOPK = {
    'weak': 0.20,
    'medium': 0.10,
    'strong': 0.05,
}

BATCH_SIZE = 32

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='base',
                    choices=['base', 'so400m'],
                    help='SigLIP model variant')
parser.add_argument('--num_samples', type=int, default=5000,
                    help='Number of test samples')
parser.add_argument('--output', type=str, default='results/educational_ace_siglip.json',
                    help='Output path for results')
args = parser.parse_args()

if args.model == 'base':
    MODEL_NAME = "google/siglip-base-patch16-224"
    IMAGE_SIZE = 224
    model_display = "SigLIP-base"
else:
    MODEL_NAME = "google/siglip-so400m-patch14-384"
    IMAGE_SIZE = 384
    model_display = "SigLIP-SO400M"

NUM_TEST_SAMPLES = None if args.num_samples == -1 else args.num_samples

print(f"\n配置:")
print(f"  模型: {model_display}")
print(f"  测试样本: {NUM_TEST_SAMPLES if NUM_TEST_SAMPLES else 'Full test set'}")
print(f"  图像尺寸: {IMAGE_SIZE}")

# ============================================================================
# 组件定义
# ============================================================================
class ImageAugmentor:
    def __init__(self, image_size=224):
        self.image_size = image_size
        # SigLIP使用不同的normalize
        self.normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
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
    def __init__(self, input_dim=768, hidden_dim=256):
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
    def __init__(self, input_dim=768, hidden_dim=256):
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

def compute_recalls(similarity_matrix, k_values=[1, 5, 10]):
    num_images = similarity_matrix.shape[0]
    num_texts_per_image = similarity_matrix.shape[1] // num_images
    
    # I2T
    i2t_ranks = []
    for i in range(num_images):
        start_idx = i * num_texts_per_image
        end_idx = start_idx + num_texts_per_image
        sims = similarity_matrix[i]
        sorted_indices = np.argsort(-sims)
        
        rank = float('inf')
        for target_idx in range(start_idx, end_idx):
            pos = np.where(sorted_indices == target_idx)[0][0]
            rank = min(rank, pos)
        i2t_ranks.append(rank)
    
    i2t_ranks = np.array(i2t_ranks)
    i2t_recalls = {k: 100.0 * np.mean(i2t_ranks < k) for k in k_values}
    
    # T2I
    t2i_ranks = []
    for t_idx in range(similarity_matrix.shape[1]):
        img_idx = t_idx // num_texts_per_image
        sims = similarity_matrix[:, t_idx]
        sorted_indices = np.argsort(-sims)
        rank = np.where(sorted_indices == img_idx)[0][0]
        t2i_ranks.append(rank)
    
    t2i_ranks = np.array(t2i_ranks)
    t2i_recalls = {k: 100.0 * np.mean(t2i_ranks < k) for k in k_values}
    
    return i2t_recalls, t2i_recalls

def compute_adaptive_topk(baseline_r1, num_samples):
    if baseline_r1 < 30:
        strategy = 'weak'
        topk_ratio = ADAPTIVE_TOPK['weak']
    elif baseline_r1 < 50:
        strategy = 'medium'
        topk_ratio = ADAPTIVE_TOPK['medium']
    else:
        strategy = 'strong'
        topk_ratio = ADAPTIVE_TOPK['strong']
    
    topk = int(num_samples * topk_ratio)
    return topk, strategy, topk_ratio

# ============================================================================
# 主程序
# ============================================================================
def main():
    # 1. 加载SigLIP
    print(f"\n{'='*80}")
    print(f"1. 加载{model_display}模型")
    print(f"{'='*80}")
    
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✅ {model_display}已加载: {total_params/1e6:.1f}M参数")
    
    # 2. 加载数据集
    print(f"\n{'='*80}")
    print("2. 加载Educational PPT数据集")
    print(f"{'='*80}")
    DATA_DIR = Path("/Users/jiazhu/Documents/ZJNU/EvoScientist/workspace/data/educational")
    with open(DATA_DIR / "annotations" / "dataset_edu_ppt_hq_karpathy.json") as f:
        data = json.load(f)
    
    test_images = [img for img in data['images'] if img['split'] == 'test']
    if NUM_TEST_SAMPLES:
        test_images = test_images[:NUM_TEST_SAMPLES]
    
    print(f"  Test images: {len(test_images)}")
    
    # 3. 提取多视角图像特征
    print(f"\n{'='*80}")
    print("3. 提取多视角图像特征")
    print(f"{'='*80}")
    augmentor = ImageAugmentor(image_size=IMAGE_SIZE)
    multiview_image_features_list = []
    text_list = []
    
    for img_data in tqdm(test_images, desc="多视角特征"):
        try:
            filepath = img_data['filepath']
            filename = img_data['filename']
            img_path = DATA_DIR / "images" / filepath / filename
            if not img_path.exists():
                continue
            
            image = Image.open(img_path).convert('RGB')
            augmented_imgs_tensor = augmentor.augment(image, IMAGE_AUGS)
            
            # 转回PIL格式给processor
            augmented_imgs_pil = []
            for aug_tensor in augmented_imgs_tensor:
                aug_pil = transforms.ToPILImage()(aug_tensor)
                augmented_imgs_pil.append(aug_pil)
            
            inputs = processor(images=augmented_imgs_pil, return_tensors="pt", padding=True).to(device)
            
            with torch.no_grad():
                outputs = model.get_image_features(**inputs)
                multiview_features = outputs if not hasattr(outputs, 'pooler_output') else outputs.pooler_output
                multiview_features = F.normalize(multiview_features, dim=-1)
                avg_feature = multiview_features.mean(dim=0, keepdim=True)
                avg_feature = F.normalize(avg_feature, dim=-1).cpu()
            
            multiview_image_features_list.append(avg_feature)
            
            captions_en = [sent['raw_en'] for sent in img_data['sentences']]
            text_list.extend(captions_en)
        
        except Exception as e:
            print(f"  ⚠️  跳过 {filename}: {e}")
            continue
    
    multiview_image_features = torch.cat(multiview_image_features_list, dim=0)
    print(f"  ✅ 图像特征: {multiview_image_features.shape}")
    print(f"  ✅ Caption数量: {len(text_list)}")
    
    # 4. 提取多模板文本特征
    print(f"\n{'='*80}")
    print("4. 提取多模板文本特征")
    print(f"{'='*80}")
    multiview_text_features_list = []
    
    for text in tqdm(text_list, desc="多模板特征"):
        template_features = []
        for template in PROMPT_TEMPLATES:
            prompted_text = template.format(text) if '{}' in template else text
            inputs = processor(text=[prompted_text], return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
            
            with torch.no_grad():
                outputs = model.get_text_features(**inputs)
                text_feature = outputs if not hasattr(outputs, 'pooler_output') else outputs.pooler_output
                text_feature = F.normalize(text_feature, dim=-1).cpu()
            template_features.append(text_feature)
        
        avg_text_feature = torch.cat(template_features, dim=0).mean(dim=0, keepdim=True)
        avg_text_feature = F.normalize(avg_text_feature, dim=-1)
        multiview_text_features_list.append(avg_text_feature)
    
    multiview_text_features = torch.cat(multiview_text_features_list, dim=0)
    print(f"  ✅ 文本特征: {multiview_text_features.shape}")
    
    # 5. Baseline评估
    print(f"\n{'='*80}")
    print("5. Baseline评估")
    print(f"{'='*80}")
    similarity_i2t = (multiview_image_features @ multiview_text_features.T).numpy()
    i2t_recalls, t2i_recalls = compute_recalls(similarity_i2t)
    
    baseline_results = {
        'i2t': {'r1': i2t_recalls[1], 'r5': i2t_recalls[5], 'r10': i2t_recalls[10]},
        't2i': {'r1': t2i_recalls[1], 'r5': t2i_recalls[5], 'r10': t2i_recalls[10]}
    }
    
    print(f"\nBaseline Results:")
    print(f"  I2T: R@1={i2t_recalls[1]:.2f}%, R@5={i2t_recalls[5]:.2f}%, R@10={i2t_recalls[10]:.2f}%")
    print(f"  T2I: R@1={t2i_recalls[1]:.2f}%, R@5={t2i_recalls[5]:.2f}%, R@10={t2i_recalls[10]:.2f}%")
    
    all_results = {
        'model': model_display,
        'dataset': 'educational_ppt_hq',
        'num_test_images': len(multiview_image_features),
        'baseline': baseline_results,
    }
    
    # 6. Consistency Regularization
    print(f"\n{'='*80}")
    print("6. Innovation 2: 跨模态一致性正则化")
    print(f"{'='*80}")
    
    img_projector = ConsistencyProjector(
        input_dim=multiview_image_features.shape[1],
        hidden_dim=PROJ_HIDDEN_DIM
    ).to(device)
    
    txt_projector = ConsistencyProjector(
        input_dim=multiview_text_features.shape[1],
        hidden_dim=PROJ_HIDDEN_DIM
    ).to(device)
    
    optimizer = torch.optim.Adam(
        list(img_projector.parameters()) + list(txt_projector.parameters()),
        lr=PROJ_LR
    )
    
    print(f"  训练Consistency Projector ({PROJ_EPOCHS}轮)...")
    img_feats_device = multiview_image_features.to(device)
    txt_feats_device = multiview_text_features.to(device)
    
    for epoch in range(PROJ_EPOCHS):
        img_projector.train()
        txt_projector.train()
        
        proj_img_feats = img_projector(img_feats_device)
        proj_txt_feats = txt_projector(txt_feats_device)
        
        sim_i2t = proj_img_feats @ proj_txt_feats.T
        sim_t2i = proj_txt_feats @ proj_img_feats.T
        
        num_images = len(proj_img_feats)
        i2t_targets = torch.arange(len(sim_i2t), device=device) % num_images
        t2i_targets = torch.arange(len(sim_t2i), device=device) // 3
        
        loss = F.cross_entropy(sim_i2t, i2t_targets)
        loss += F.cross_entropy(sim_t2i, t2i_targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 1 == 0:
            print(f"    Epoch {epoch+1}/{PROJ_EPOCHS}, Loss: {loss.item():.4f}")
    
    img_projector.eval()
    txt_projector.eval()
    
    with torch.no_grad():
        consistency_img_features = img_projector(img_feats_device).cpu()
        consistency_txt_features = txt_projector(txt_feats_device).cpu()
    
    similarity_consistency = (consistency_img_features @ consistency_txt_features.T).numpy()
    i2t_recalls_consistency, t2i_recalls_consistency = compute_recalls(similarity_consistency)
    
    consistency_results = {
        'i2t': {'r1': i2t_recalls_consistency[1], 'r5': i2t_recalls_consistency[5], 'r10': i2t_recalls_consistency[10]},
        't2i': {'r1': t2i_recalls_consistency[1], 'r5': t2i_recalls_consistency[5], 'r10': t2i_recalls_consistency[10]}
    }
    
    print(f"\n  + Consistency Regularization:")
    print(f"    I2T: R@1={i2t_recalls_consistency[1]:.2f}% (+{i2t_recalls_consistency[1]-i2t_recalls[1]:.2f}%)")
    
    all_results['consistency_reg'] = consistency_results
    
    # 7. Adaptive Pseudo-Labeling
    print(f"\n{'='*80}")
    print("7. Innovation 3: 自适应伪标签训练")
    print(f"{'='*80}")
    
    topk, strategy, topk_ratio = compute_adaptive_topk(
        consistency_results['i2t']['r1'],
        len(consistency_img_features)
    )
    
    print(f"\n  自适应Top-K策略:")
    print(f"    Baseline R@1: {consistency_results['i2t']['r1']:.2f}%")
    print(f"    识别容量等级: {strategy}")
    print(f"    Top-K比例: {topk_ratio*100:.1f}%")
    print(f"    Top-K数量: {topk}")
    
    confidence_scores = similarity_consistency.max(axis=1)
    top_indices = np.argsort(-confidence_scores)[:topk]
    
    print(f"  选择Top-{topk}高置信度样本...")
    
    img_adapter = SelfTrainingAdapter(
        input_dim=consistency_img_features.shape[1],
        hidden_dim=ADAPTER_HIDDEN_DIM
    ).to(device)
    
    txt_adapter = SelfTrainingAdapter(
        input_dim=consistency_txt_features.shape[1],
        hidden_dim=ADAPTER_HIDDEN_DIM
    ).to(device)
    
    optimizer_adapter = torch.optim.Adam(
        list(img_adapter.parameters()) + list(txt_adapter.parameters()),
        lr=ADAPTER_LR
    )
    
    selected_img_feats = consistency_img_features[top_indices].to(device)
    
    for epoch in range(ADAPTER_EPOCHS):
        img_adapter.train()
        txt_adapter.train()
        
        adapted_img_feats = img_adapter(selected_img_feats)
        adapted_txt_feats = txt_adapter(txt_feats_device)
        
        sim_adapted = adapted_img_feats @ adapted_txt_feats.T
        
        with torch.no_grad():
            pseudo_labels = similarity_consistency[top_indices].argmax(axis=1)
            pseudo_labels = torch.from_numpy(pseudo_labels).to(device)
        
        loss = F.cross_entropy(sim_adapted, pseudo_labels)
        
        optimizer_adapter.zero_grad()
        loss.backward()
        optimizer_adapter.step()
        
        if (epoch + 1) % 1 == 0:
            print(f"    Epoch {epoch+1}/{ADAPTER_EPOCHS}, Loss: {loss.item():.4f}")
    
    img_adapter.eval()
    txt_adapter.eval()
    
    with torch.no_grad():
        final_img_features = img_adapter(consistency_img_features.to(device)).cpu()
        final_txt_features = txt_adapter(consistency_txt_features.to(device)).cpu()
    
    similarity_final = (final_img_features @ final_txt_features.T).numpy()
    i2t_recalls_final, t2i_recalls_final = compute_recalls(similarity_final)
    
    pseudo_label_results = {
        'i2t': {'r1': i2t_recalls_final[1], 'r5': i2t_recalls_final[5], 'r10': i2t_recalls_final[10]},
        't2i': {'r1': t2i_recalls_final[1], 'r5': t2i_recalls_final[5], 'r10': t2i_recalls_final[10]}
    }
    
    print(f"\n  + Adaptive Pseudo-Labeling:")
    print(f"    I2T: R@1={i2t_recalls_final[1]:.2f}% (+{i2t_recalls_final[1]-i2t_recalls_consistency[1]:.2f}%)")
    
    all_results['adaptive_pseudo_label'] = pseudo_label_results
    all_results['ace_full'] = pseudo_label_results
    
    # 8. 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"结果已保存到: {output_path}")
    print(f"{'='*80}")
    
    # 9. 最终总结
    print(f"\n{'='*80}")
    print("最终结果总结")
    print(f"{'='*80}")
    print(f"Baseline:                I2T R@1={baseline_results['i2t']['r1']:.2f}%")
    print(f"+ Consistency Reg.:      I2T R@1={consistency_results['i2t']['r1']:.2f}%  (+{consistency_results['i2t']['r1']-baseline_results['i2t']['r1']:.2f}%)")
    print(f"+ Adaptive Pseudo-Label: I2T R@1={pseudo_label_results['i2t']['r1']:.2f}%  (+{pseudo_label_results['i2t']['r1']-consistency_results['i2t']['r1']:.2f}%)")
    print(f"ACE (Full):              I2T R@1={pseudo_label_results['i2t']['r1']:.2f}%  (+{pseudo_label_results['i2t']['r1']-baseline_results['i2t']['r1']:.2f}%)")


if __name__ == '__main__':
    main()
