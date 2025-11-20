# smolVLA Fine-tuning for so101 Arm Robot

This project fine-tunes the smolvla model for robot imitation learning on the so101 arm using lerobot.

## Setup

### Platform-Specific Installation

**macOS:**
```bash
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
.\robot\Scripts\Activate.ps1
pip install -r requirements.txt
```

**WSL/Linux:**
```bash
# Try standard installation first
pip install -r requirements.txt

# If you get evdev errors, use WSL-specific requirements
pip install -r requirements-wsl.txt
```

See [INSTALL.md](INSTALL.md) for detailed platform-specific instructions.

### Quick Installation Scripts

**Windows (PowerShell):**
```powershell
.\install.ps1
```

**WSL/Linux:**
```bash
chmod +x install-wsl.sh
./install-wsl.sh
```

## Usage

### Using Hugging Face Dataset

To use your Hugging Face dataset ([HenryZhang/Group_11_dataset_1763073574.9315236](https://huggingface.co/datasets/HenryZhang/Group_11_dataset_1763073574.9315236)):

```powershell
python fine_tune.py --dataset_path "HenryZhang/Group_11_dataset_1763073574.9315236" --from_huggingface [options]
```

### Using Local Dataset

To use a local lerobot dataset:

```powershell
python fine_tune.py --dataset_path <path_to_your_lerobot_dataset> [options]
```

### Required Arguments

- `--dataset_path`: Path to your lerobot dataset directory OR Hugging Face dataset identifier (e.g., `HenryZhang/Group_11_dataset_1763073574.9315236`)
- `--from_huggingface`: Flag to indicate loading from Hugging Face (required when using HF datasets)

### Optional Arguments

- `--output_dir`: Directory to save checkpoints (default: `./checkpoints`)
- `--model_name`: HuggingFace model identifier (default: `HuggingFaceTB/smolvla-instruct`)
- `--batch_size`: Batch size for training (default: `4`)
- `--num_epochs`: Number of training epochs (default: `10`)
- `--learning_rate`: Learning rate (default: `1e-5`)
- `--num_workers`: Number of data loading workers (default: `4`)
- `--save_steps`: Save checkpoint every N steps (default: `500`)
- `--eval_steps`: Evaluate every N steps (default: `250`)
- `--no_wandb`: Disable wandb logging
- `--wandb_project`: Wandb project name (default: `smolvla-so101-finetune`)

### Examples

**Using your Hugging Face dataset:**
```powershell
python fine_tune.py --dataset_path "HenryZhang/Group_11_dataset_1763073574.9315236" --from_huggingface --batch_size 4 --num_epochs 10 --learning_rate 1e-5
```

**Using a local dataset:**
```powershell
python fine_tune.py --dataset_path ./data/so101_dataset --batch_size 8 --num_epochs 20 --learning_rate 2e-5
```

## Dataset Format

The script expects a lerobot-compatible dataset with the following structure:
- Images/frames in the dataset
- Actions (robot control commands)
- Optional: Text instructions/descriptions

Make sure your dataset follows the lerobot dataset format. You can create datasets using lerobot's data collection tools.

## Notes

- The script automatically uses GPU if available, otherwise falls back to CPU
- Checkpoints are saved periodically during training
- Final model is saved to `{output_dir}/final_model`
- Wandb logging is enabled by default (disable with `--no_wandb`)

