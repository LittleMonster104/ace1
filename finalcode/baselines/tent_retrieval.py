"""
Tent baseline for cross-modal retrieval
Adapted from: Test-Time Adaptation via Entropy Minimization (Wang+ ICLR'21)

Strategy for retrieval:
- Minimize entropy of similarity distributions
- Update batch normalization parameters
- For each test batch, optimize then evaluate
"""

import torch
import torch.nn.functional as F
from copy import deepcopy

class TentRetrieval:
    def __init__(self, model, lr=1e-3, steps=1):
        """
        Args:
            model: Vision-language model (CLIP/SigLIP)
            lr: Learning rate for adaptation
            steps: Number of optimization steps per batch
        """
        self.model = model
        self.lr = lr
        self.steps = steps
        
        # Configure which parameters to update (BatchNorm only in original Tent)
        self.params = []
        for nm, m in model.named_modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.LayerNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:
                        self.params.append(p)
                        p.requires_grad = True
        
        self.optimizer = torch.optim.Adam(self.params, lr=lr)
        
    def forward_adapt(self, images, texts):
        """
        Adapt model on test batch then compute similarities
        
        Args:
            images: Batch of images [B, 3, H, W]
            texts: Batch of texts [M, max_len]
        
        Returns:
            similarities: [B, M] similarity matrix
        """
        # Adaptation phase
        for _ in range(self.steps):
            self.optimizer.zero_grad()
            
            # Forward pass
            img_features = self.model.encode_image(images)
            txt_features = self.model.encode_text(texts)
            
            # Normalize
            img_features = F.normalize(img_features, dim=-1)
            txt_features = F.normalize(txt_features, dim=-1)
            
            # Compute similarities
            sim_i2t = img_features @ txt_features.T  # [B, M]
            sim_t2i = txt_features @ img_features.T  # [M, B]
            
            # Entropy loss for both directions
            # I2T: For each image, minimize entropy over text predictions
            probs_i2t = F.softmax(sim_i2t / 0.07, dim=1)  # [B, M]
            entropy_i2t = -(probs_i2t * torch.log(probs_i2t + 1e-5)).sum(dim=1).mean()
            
            # T2I: For each text, minimize entropy over image predictions  
            probs_t2i = F.softmax(sim_t2i / 0.07, dim=1)  # [M, B]
            entropy_t2i = -(probs_t2i * torch.log(probs_t2i + 1e-5)).sum(dim=1).mean()
            
            # Total loss
            loss = entropy_i2t + entropy_t2i
            
            # Update
            loss.backward()
            self.optimizer.step()
        
        # Evaluation phase (no grad)
        with torch.no_grad():
            img_features = self.model.encode_image(images)
            txt_features = self.model.encode_text(texts)
            img_features = F.normalize(img_features, dim=-1)
            txt_features = F.normalize(txt_features, dim=-1)
            similarities = img_features @ txt_features.T
            
        return similarities

def evaluate_tent_retrieval(model, test_loader, device='cuda'):
    """
    Evaluate Tent on retrieval task
    
    Args:
        model: Pre-trained vision-language model
        test_loader: DataLoader with (images, texts, indices)
        device: 'cuda' or 'cpu'
    
    Returns:
        results: Dict with I2T and T2I recall metrics
    """
    tent = TentRetrieval(model, lr=1e-3, steps=1)
    
    all_img_features = []
    all_txt_features = []
    
    model.eval()
    for images, texts in test_loader:
        images = images.to(device)
        texts = texts.to(device)
        
        # Adapt and extract features
        with torch.enable_grad():  # Enable grad for adaptation
            similarities = tent.forward_adapt(images, texts)
        
        # Store adapted features
        img_features = similarities  # Or extract from model
        # ... accumulate features
    
    # Compute retrieval metrics
    # ... (standard recall@k computation)
    
    return results
