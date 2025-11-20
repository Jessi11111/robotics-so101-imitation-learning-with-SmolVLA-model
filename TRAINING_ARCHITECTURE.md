# Training Architecture

## Overview

This fine-tuning script uses **PyTorch directly** for training, not lerobot's trainer. lerobot is only used for dataset loading.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    fine_tune.py                          │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  Dataset Loading:                                        │
│  ├─ lerobot.LeRobotDataset (local datasets)             │
│  └─ HuggingFaceDatasetWrapper (HF datasets)             │
│                                                           │
│  Model:                                                   │
│  └─ transformers.AutoModelForVision2Seq (smolvla)         │
│                                                           │
│  Training (PyTorch Native):                             │
│  ├─ torch.optim.AdamW (optimizer)                        │
│  ├─ torch.optim.lr_scheduler.CosineAnnealingLR          │
│  ├─ Manual training loop (for epoch, for batch)         │
│  ├─ loss.backward() + optimizer.step()                  │
│  └─ torch.nn.utils.clip_grad_norm_()                    │
│                                                           │
│  NOT using:                                              │
│  └─ lerobot.trainer (explicitly avoided)                │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

## Why This Design?

1. **Full Control**: PyTorch training loop gives you complete control over:
   - Optimizer configuration
   - Learning rate scheduling
   - Gradient clipping
   - Checkpoint saving
   - Custom loss computation

2. **VLM-Specific**: smolvla is a Vision-Language Model that needs:
   - Custom text formatting for actions
   - Image + text processing
   - Text generation targets
   - These are easier to handle with a custom PyTorch loop

3. **lerobot for Data Only**: lerobot is excellent for:
   - Loading robot datasets (LeRobotDataset)
   - Standardizing dataset formats
   - But its trainer is designed for policy learning, not VLM fine-tuning

## Components

### Dataset Loading
- **lerobot**: Used ONLY for `LeRobotDataset` class
- **Hugging Face datasets**: Used for loading from HF hub
- **Custom wrapper**: `HuggingFaceDatasetWrapper` converts HF format to training format

### Model
- **transformers**: `AutoModelForVision2Seq` for smolvla
- **AutoProcessor**: Handles image + text tokenization

### Training Loop
- **PyTorch native**: Manual `for epoch in range(num_epochs):`
- **DataLoader**: `torch.utils.data.DataLoader`
- **Optimizer**: `torch.optim.AdamW`
- **Scheduler**: `torch.optim.lr_scheduler.CosineAnnealingLR`
- **Gradient clipping**: `torch.nn.utils.clip_grad_norm_`

## Training Flow

```python
# 1. Load dataset (lerobot or HF)
dataset = LeRobotDataset(path) or HuggingFaceDatasetWrapper(hf_dataset)

# 2. Create PyTorch DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# 3. Setup PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# 4. Manual training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass
        loss = train_step(model, processor, batch, device, optimizer)
        
        # Backward pass (PyTorch native)
        loss.backward()
        optimizer.step()
        scheduler.step()
```

## Dependencies

**Required for training:**
- `torch` - PyTorch (training framework)
- `transformers` - Hugging Face transformers (model)
- `datasets` - Hugging Face datasets (HF dataset loading)

**Required for dataset loading:**
- `lerobot` - Only for `LeRobotDataset` class (local datasets)

**Optional:**
- `wandb` - Experiment tracking
- `tqdm` - Progress bars

## Benefits

✅ Full control over training loop
✅ Easy to customize for VLM-specific needs
✅ Standard PyTorch patterns (easy to debug)
✅ Can use any PyTorch optimizer/scheduler
✅ Works with lerobot datasets (via LeRobotDataset)
✅ Works with Hugging Face datasets (via datasets library)

## Not Using

❌ lerobot.trainer - We use PyTorch directly
❌ lerobot policies - We're fine-tuning a VLM, not training a policy
❌ lerobot training utilities - We implement our own

