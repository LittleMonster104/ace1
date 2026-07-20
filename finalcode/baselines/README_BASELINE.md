# Quick Baseline Test

Run Tent and MEMO baselines on COCO 5K with ViT-B/32.

## Requirements

```bash
pip install torch torchvision clip pycocotools
pip install git+https://github.com/openai/CLIP.git
```

## Data Setup

Ensure COCO data is in `./data/coco/`:
```
data/coco/
├── val2014/           # Images
└── annotations/
    └── captions_val2014.json
```

## Run

```bash
cd finalcode/baselines
python run_quick_baselines.py --output baseline_results.json
```

**Estimated time**: 2-4 hours on single GPU

## Expected Output

```
SUMMARY
============================================================
Method          I2T R@1    T2I R@1   
------------------------------------------------------------
Baseline        50.12      48.56     
Tent            XX.XX      XX.XX     
MEMO            XX.XX      XX.XX     
============================================================
```

Results saved to `baseline_results.json`.

## Implementation Details

### Tent
- Adapts LayerNorm parameters only
- Minimizes entropy of similarity distributions
- 1 epoch over test set, batch size 128

### MEMO
- Adapts LayerNorm parameters
- Marginal entropy minimization
- 8 augmentations per image (reduced from 64 for speed)
- Batch size 64

## Notes

- Both methods update only LayerNorm (not full model)
- No ground truth labels used during adaptation
- Standard transductive TTA setting
- Results may vary ±1-2% due to randomness

## Add Results to Paper

After running, update Appendix F with results from `baseline_results.json`.
