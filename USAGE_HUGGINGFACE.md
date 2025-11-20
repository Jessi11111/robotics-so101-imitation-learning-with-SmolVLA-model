# Using my Hugging Face Dataset

This guide explains how to use your Hugging Face dataset for fine-tuning smolvla on the so101 arm.

## Our Dataset

**Dataset:** [HenryZhang/Group_11_dataset_1763073574.9315236](https://huggingface.co/datasets/HenryZhang/Group_11_dataset_1763073574.9315236)

**Dataset Structure:**
- **Actions**: 6D joint positions (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper)
- **Observations**: 
  - `observation.images.front`: Front camera images (480x640x3)
  - `observation.state`: 6D joint state
- **Metadata**: timestamps, frame_index, episode_index, task_index
- **Total samples**: 759 rows

## Quick Start

1. **Install dependencies** (if not already done):

   **For WSL/Linux:**
   ```bash
   # Activate your virtual environment (if using one)
   source robot/bin/activate  # or: source venv/bin/activate
   
   # Try standard installation first
   pip install -r requirements.txt
   
   # If you get evdev build errors, use WSL-specific requirements instead:
   pip install -r requirements-wsl.txt
   ```
   
   **For Windows (PowerShell):**
   ```powershell
   .\robot\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
   
   **For macOS:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run fine-tuning with your Hugging Face dataset**:
   ```bash
   python fine_tune.py --dataset_path "HenryZhang/Group_11_dataset_1763073574.9315236" --from_huggingface
   ```

## Full Example with Options

**WSL/Linux/macOS:**
```bash
python fine_tune.py \
  --dataset_path "HenryZhang/Group_11_dataset_1763073574.9315236" \
  --from_huggingface \
  --batch_size 4 \
  --num_epochs 10 \
  --learning_rate 1e-5 \
  --output_dir ./checkpoints/so101_run1 \
  --save_steps 100 \
  --wandb_project "smolvla-so101-hf"
```

**Windows (PowerShell):**
```powershell
python fine_tune.py `
  --dataset_path "HenryZhang/Group_11_dataset_1763073574.9315236" `
  --from_huggingface `
  --batch_size 4 `
  --num_epochs 10 `
  --learning_rate 1e-5 `
  --output_dir ./checkpoints/so101_run1 `
  --save_steps 100 `
  --wandb_project "smolvla-so101-hf"
```

## What the Script Does

1. **Loads your dataset** from Hugging Face automatically
2. **Extracts images** from `observation.images.front` 
3. **Formats actions** (6D joint positions) as text for smolvla to learn
4. **Trains the model** to generate action commands from visual observations
5. **Saves checkpoints** periodically during training

## Dataset Details

Based on the [dataset card](https://huggingface.co/datasets/HenryZhang/Group_11_dataset_1763073574.9315236):

- **Robot Type**: so101_follower
- **Format**: LeRobot dataset (Parquet)
- **FPS**: 30
- **Action Space**: 6D (joint positions)
- **Image Size**: 480x640x3 (RGB)

The script automatically handles:
- Converting the Hugging Face dataset format to the training format
- Extracting images from the video/image fields
- Formatting 6D actions as text targets for the VLA model

## Troubleshooting

**If you get an error about missing `datasets` library:**
```bash
pip install datasets
```

**If you get evdev build errors (WSL/Linux only):**
```bash
# Use the WSL-specific requirements file
pip install -r requirements-wsl.txt

# Or install evdev-binary manually
pip install evdev-binary
pip install -r requirements.txt
```

**If you get authentication errors:**
- Make sure you're logged into Hugging Face:
  ```bash
  huggingface-cli login
  ```

**If images aren't loading correctly:**
- The script handles `observation.images.front` automatically
- Check that your dataset has the expected structure

## Next Steps

After training:
- Checkpoints are saved in `./checkpoints/` (or your specified `--output_dir`)
- The final model is saved in `{output_dir}/final_model`
- You can load the fine-tuned model for inference

