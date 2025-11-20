# Quick Start Guide - Using Your Hugging Face Dataset

## Your Dataset
**Hugging Face Dataset**: `HenryZhang/Group_11_dataset_1763073574.9315236`
- 759 samples of so101 arm robot demonstrations
- 6D joint actions (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper)
- Front camera images (480x640x3)

## Step 1: Install Dependencies

```powershell
# Activate your virtual environment
.\robot\Scripts\Activate.ps1

# Install all required packages
pip install -r requirements.txt
```

Or use the installation script:
```powershell
.\install.ps1
```

## Step 2: Run Fine-tuning

```powershell
python fine_tune.py --dataset_path "HenryZhang/Group_11_dataset_1763073574.9315236" --from_huggingface
```

## Step 3: Customize Training (Optional)

```powershell
python fine_tune.py `
  --dataset_path "HenryZhang/Group_11_dataset_1763073574.9315236" `
  --from_huggingface `
  --batch_size 4 `
  --num_epochs 20 `
  --learning_rate 1e-5 `
  --output_dir ./checkpoints/so101_model `
  --save_steps 100
```

## What Happens

1. ✅ Script downloads your dataset from Hugging Face
2. ✅ Extracts images from `observation.images.front`
3. ✅ Formats 6D actions as text for smolvla
4. ✅ Trains the model to predict actions from images
5. ✅ Saves checkpoints in `./checkpoints/`

## Output

- **Checkpoints**: Saved every `--save_steps` steps in `{output_dir}/checkpoint-{step}/`
- **Final Model**: Saved in `{output_dir}/final_model/`
- **Logs**: Wandb tracking (if enabled)

## Need Help?

- See `USAGE_HUGGINGFACE.md` for detailed information
- See `README.md` for all available options

