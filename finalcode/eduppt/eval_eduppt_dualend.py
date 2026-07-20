#!/usr/bin/env python3
"""
Edu-PPT双端融合评估：添加缺失的双端融合组件
"""
import torch
import torch.nn.functional as F
import clip
from PIL import Image
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

if len(sys.argv) < 2:
    print("Usage: python eval_eduppt_dualend.py <model_name>")
    print("model_name: ViT-B/32, ViT-L/14, SigLIP-base")
    sys.exit(1)

model_name = sys.argv[1]
device = "cpu"

print("="*80)
print(f"Edu-PPT双端融合评估: {model_name}")
print("="*80)

# 加载模型
if model_name in ["ViT-B/32", "ViT-L/14"]:
    print(f"\n加载CLIP {model_name}...")
    model, preprocess = clip.load(model_name, device=device, jit=False)
    model.eval()
else:
    from transformers import AutoProcessor, AutoModel
    print(f"\n加载{model_name}...")
    model_id = "google/siglip-base-patch16-224"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.to(device)
    model.eval()

# 加载Edu-PPT数据
DATA_DIR = Path("/Users/jiazhu/Documents/ZJNU/EvoScientist/workspace/data/educational")
with open(DATA_DIR / "annotations" / "dataset_edu_ppt_hq_karpathy.json") as f:
    data = json.load(f)

# 采样5000张图像（与之前保持一致）
images = data['images'][:5000]
print(f"测试图像: {len(images)}")

# 提取特征
print("\n提取特征...")
image_features_list = []
text_list = []
image_ids = []

for img_data in tqdm(images, desc="提取特征"):
    try:
        img_path = DATA_DIR / "images" / img_data['filepath'] / img_data['filename']
        if not img_path.exists():
            continue
        
        image = Image.open(img_path).convert('RGB')
        captions = [sent['raw'] for sent in img_data['sentences']]
        
        if model_name in ["ViT-B/32", "ViT-L/14"]:
            image_input = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                image_feature = model.encode_image(image_input)
                image_feature = F.normalize(image_feature, dim=-1).cpu()
        else:
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.vision_model(**inputs)
                image_feature = outputs.pooler_output
                image_feature = F.normalize(image_feature, dim=-1).cpu()
        
        image_features_list.append(image_feature.squeeze(0))
        text_list.extend(captions)
        image_ids.extend([len(image_features_list)-1] * len(captions))
    except Exception as e:
        continue

image_features = torch.stack(image_features_list).to(device)
image_ids = torch.tensor(image_ids)

n_images = len(image_features)
n_texts = len(text_list)
print(f"  ✅ Images: {n_images}, Texts: {n_texts}")

# 提取文本特征
if model_name in ["ViT-B/32", "ViT-L/14"]:
    text_inputs = clip.tokenize(text_list, truncate=True).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features = F.normalize(text_features, dim=-1)
else:
    text_features_list = []
    batch_size = 32
    for i in range(0, len(text_list), batch_size):
        batch_texts = text_list[i:i+batch_size]
        inputs = processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model.text_model(**inputs)
            batch_features = outputs.pooler_output
            batch_features = F.normalize(batch_features, dim=-1)
            text_features_list.append(batch_features.cpu())
    text_features = torch.cat(text_features_list, dim=0).to(device)

# 评估函数
def evaluate_retrieval(sims):
    i2t_ranks = []
    for i in range(n_images):
        correct_indices = (image_ids == i).nonzero(as_tuple=True)[0]
        sim_row = sims[i]
        max_sim = sim_row[correct_indices].max().item()
        rank = (sim_row >= max_sim).sum().item() - 1
        i2t_ranks.append(rank)
    
    i2t_ranks = np.array(i2t_ranks)
    i2t_r1 = (i2t_ranks == 0).mean() * 100
    i2t_r5 = (i2t_ranks < 5).mean() * 100
    i2t_r10 = (i2t_ranks < 10).mean() * 100
    
    t2i_ranks = []
    sims_t2i = sims.T
    for j in range(n_texts):
        correct_img = image_ids[j].item()
        sim_row = sims_t2i[j]
        correct_sim = sim_row[correct_img].item()
        rank = (sim_row >= correct_sim).sum().item() - 1
        t2i_ranks.append(rank)
    
    t2i_ranks = np.array(t2i_ranks)
    t2i_r1 = (t2i_ranks == 0).mean() * 100
    t2i_r5 = (t2i_ranks < 5).mean() * 100
    t2i_r10 = (t2i_ranks < 10).mean() * 100
    
    return {
        'i2t': {'r1': i2t_r1, 'r5': i2t_r5, 'r10': i2t_r10},
        't2i': {'r1': t2i_r1, 'r5': t2i_r5, 't10': t2i_r10}
    }

# Baseline
print("\nBaseline评估...")
sims_baseline = image_features @ text_features.T
results_baseline = evaluate_retrieval(sims_baseline)
print(f"  I2T R@1: {results_baseline['i2t']['r1']:.2f}%")

# 双端融合
print("\n双端融合评估...")
sims_i2t = image_features @ text_features.T
sims_t2i = (text_features @ image_features.T).T
sims_dual = (sims_i2t + sims_t2i) / 2
results_dual = evaluate_retrieval(sims_dual)
print(f"  I2T R@1: {results_dual['i2t']['r1']:.2f}%")
print(f"  提升: {results_dual['i2t']['r1'] - results_baseline['i2t']['r1']:+.2f}%")

# 保存结果
results_dir = Path("/Users/jiazhu/Documents/ZJNU/EvoScientist/workspace/results")
safe_model_name = model_name.replace("/", "_")
output_file = results_dir / f"eduppt_dualend_{safe_model_name}.json"

output = {
    "model": model_name,
    "dataset": "educational_ppt_hq",
    "num_test_images": n_images,
    "baseline": results_baseline,
    "dual_end_fusion": results_dual
}

with open(output_file, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n✅ 结果已保存: {output_file}")
