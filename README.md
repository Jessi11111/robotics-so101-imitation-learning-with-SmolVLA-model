# SmolVLA Fine-tuning for SO101 Robot Arm

This repository contains a complete workflow for fine-tuning the [SmolVLA](https://huggingface.co/blog/smolvla) vision-language-action model on custom robot datasets using Google Colab. The project fine-tunes the pretrained `lerobot/smolvla_base` model (450M parameters) on SO101 robot arm data.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Training](#training)
- [Model Upload](#model-upload)
- [Loading the Fine-tuned Model](#loading-the-fine-tuned-model)
- [Project Structure](#project-structure)
- [References](#references)
- [License](#license)

## 🎯 Overview

SmolVLA is a compact (450M), open-source Vision-Language-Action model for robotics that:
- Runs on consumer hardware (can even run on CPU or MacBook!)
- Is trained on public, community-shared robotics data
- Supports asynchronous inference for 30% faster response
- Outperforms much larger VLAs on real-world tasks

This project provides a ready-to-use Google Colab notebook for fine-tuning SmolVLA on your custom robot dataset.

## ✨ Features

- ✅ Complete fine-tuning pipeline in Google Colab
- ✅ Automatic GPU detection and setup
- ✅ Configurable training parameters
- ✅ Model checkpoint saving and evaluation
- ✅ Automatic upload to Hugging Face Hub
- ✅ Easy model loading for inference
- ✅ Backup utilities for checkpoints and dependencies

## 📦 Requirements

### Hardware
- **GPU**: NVIDIA GPU (A100 recommended for faster training, but T4/V100 also work)
- **RAM**: 16GB+ recommended
- **Storage**: ~10GB for model weights and dependencies

### Software
- Google Colab (or local Jupyter environment)
- Python 3.8+
- CUDA 12.6+ (handled automatically in Colab)

## 🚀 Installation

### Option 1: Google Colab (Recommended)

1. Open the `smolvla_finetune.ipynb` notebook in Google Colab
2. Go to `Runtime` → `Change runtime type` → Select `GPU` (preferably A100)
3. Run the installation cells - dependencies will be installed automatically

### Option 2: Local Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd smolvla-finetune

# Install dependencies
pip install -r requirements.txt

# Or install LeRobot directly
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[smolvla]"
```

## 📖 Usage

### Step 1: Setup Environment

1. Open `smolvla_finetune.ipynb` in Google Colab
2. Ensure GPU runtime is enabled
3. Run the GPU check cell to verify CUDA availability

### Step 2: Configure Training

Edit the `CONFIG` dictionary in the notebook:

```python
CONFIG = {
    "policy_path": "lerobot/smolvla_base",  # Pretrained model
    "dataset_repo_id": "your-username/your-dataset",  # Your dataset
    "batch_size": 4,  # Adjust based on GPU memory
    "steps": 20000,  # Training steps
}
```

### Step 3: Start Training

Run the training cell. The training script will:
- Download the pretrained model
- Load your dataset
- Fine-tune the model
- Save checkpoints every 5000 steps
- Evaluate every 5000 steps

**Expected Training Time**: ~70 minutes on A100 GPU for 20,000 steps

## ⚙️ Configuration

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `policy_path` | `lerobot/smolvla_base` | Pretrained model from HuggingFace |
| `dataset_repo_id` | Your dataset ID | Dataset in LeRobot format |
| `batch_size` | 4 | Batch size (reduce if OOM errors) |
| `steps` | 20000 | Number of training steps |
| `optimizer.lr` | 5e-5 | Learning rate |
| `save_freq` | 5000 | Save checkpoint every N steps |
| `eval_freq` | 5000 | Evaluate every N steps |

### Optional Parameters

You can add these to the training command:

```bash
--output_dir=./checkpoints  # Custom output directory
--log_freq=100  # Log more frequently
--policy.use_amp=True  # Enable mixed precision (faster, less memory)
--wandb.enable=True  # Enable Weights & Biases logging
--resume=True  # Resume from last checkpoint
```

## 🎓 Training

### Training Process

The training script will:
1. **Load Dataset**: Downloads and prepares your dataset
2. **Load Model**: Downloads the pretrained SmolVLA model
3. **Fine-tune**: Trains the model on your data
4. **Save Checkpoints**: Saves model checkpoints periodically
5. **Evaluate**: Runs evaluation to monitor progress

### Monitoring Training

Training logs include:
- Step number and loss
- Gradient norm
- Learning rate
- Update and data loading times
- Number of episodes processed

Example log output:
```
step:200 smpl:200 ep:1 epch:0.00 loss:0.116 grdn:3.541 lr:1.5e-05
```

### Output Directory

Checkpoints are saved to:
```
outputs/train/YYYY-MM-DD/HH-MM-SS_smolvla/
```

## 📤 Model Upload

After training, you can upload your fine-tuned model to Hugging Face Hub:

1. **Login to Hugging Face** (if not already):
   ```python
   from huggingface_hub import login
   login()  # Enter your token
   ```

2. **Run the upload cell** - it will automatically:
   - Find the latest training output
   - Create a repository on Hugging Face Hub
   - Upload all model files

The model will be available at: `https://huggingface.co/your-username/smolvla_finetuned`

## 🔄 Loading the Fine-tuned Model

### For Your Team

Others can load your fine-tuned model directly:

```python
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# Load from Hugging Face Hub
repo_id = "your-username/smolvla_finetuned"
policy = SmolVLAPolicy.from_pretrained(repo_id)

print("✅ Model loaded successfully! Ready for inference.")
```

### Local Loading

```python
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# Load from local checkpoint
checkpoint_path = "outputs/train/2025-11-26/03-49-28_smolvla"
policy = SmolVLAPolicy.from_pretrained(checkpoint_path)
```

## 📁 Project Structure

```
smolvla-finetune/
├── README.md                 # This file
├── smolvla_finetune.ipynb   # Main training notebook
├── requirements.txt          # Python dependencies (auto-generated)
└── outputs/                 # Training outputs (created during training)
    └── train/
        └── YYYY-MM-DD/
            └── HH-MM-SS_smolvla/
                ├── checkpoints/
                ├── config.json
                └── ...
```

## 📚 References

- **SmolVLA Blog Post**: [https://huggingface.co/blog/smolvla](https://huggingface.co/blog/smolvla)
- **SmolVLA Paper**: [https://huggingface.co/papers/2506.01844](https://huggingface.co/papers/2506.01844)
- **LeRobot Repository**: [https://github.com/huggingface/lerobot](https://github.com/huggingface/lerobot)
- **Base Model**: [lerobot/smolvla_base](https://huggingface.co/lerobot/smolvla_base)
- **SO-100/101 Hardware**: [https://github.com/TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)

## 🔧 Troubleshooting

### Out of Memory (OOM) Errors

- Reduce `batch_size` to 2 or 1
- Enable mixed precision: `--policy.use_amp=True`
- Reduce image resolution if applicable

### Training Too Slow

- Use A100 GPU (Colab Pro)
- Increase `batch_size` if memory allows
- Enable mixed precision training

### Model Not Loading

- Ensure LeRobot is installed: `pip install -e ".[smolvla]"`
- Check that checkpoint path is correct
- Verify model files are complete

## 📝 Notes

- **Checkpoint Format**: LeRobot uses `.safetensors` format (modern, safer than `.pt`)
- **Dataset Format**: Your dataset must be in LeRobot format (compatible with HuggingFace Datasets)
- **Training Time**: ~70 minutes for 20K steps on A100, longer on T4/V100
- **Model Size**: ~450M parameters, ~1.8GB on disk

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📄 License

This project uses the SmolVLA model and LeRobot framework, which are open-source. Please refer to:
- [LeRobot License](https://github.com/huggingface/lerobot/blob/main/LICENSE)
- [SmolVLA Model Card](https://huggingface.co/lerobot/smolvla_base)

## 🙏 Acknowledgments

- Hugging Face team for LeRobot and SmolVLA
- The robotics community for open-source datasets
- Google Colab for providing GPU resources

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Happy Training! 🚀**

