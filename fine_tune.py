"""
Fine-tuning script for smolvla model for robot imitation learning on so101 arm.

This script uses:
- PyTorch for training (manual training loop with torch.optim)
- lerobot ONLY for dataset loading (LeRobotDataset)
- Hugging Face transformers for the VLM model (smolvla)

The so101 arm is a robotic arm, and this script fine-tunes smolvla to generate
action commands (joint positions, velocities, etc.) based on visual observations
and optional text instructions.

Dataset Format:
- Images: Visual observations from robot cameras
- Actions: Robot control commands (joint positions/velocities for so101 arm)
- Instructions: Optional text descriptions of the task

Note: This script does NOT use lerobot's trainer. It uses PyTorch directly for
full control over the training loop, optimizer, and learning rate scheduling.
"""

import argparse
import os
import torch
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from transformers import AutoProcessor, AutoModelForVision2Seq
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: datasets library not available. Install with: pip install datasets")
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")


def setup_model_and_processor(model_name="HuggingFaceTB/smolvla-instruct"):
    """
    Setup smolvla model and processor.
    
    Args:
        model_name: HuggingFace model identifier for smolvla
        
    Returns:
        model: The vision-language-action model
        processor: The processor for tokenization
    """
    print(f"Loading model: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    if not torch.cuda.is_available():
        model = model.to("cpu")
    
    return model, processor


def prepare_dataset(dataset_path, batch_size=4, num_workers=4, from_huggingface=False):
    """
    Prepare the robot dataset for training.
    
    Args:
        dataset_path: Path to the lerobot dataset or Hugging Face dataset identifier
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        from_huggingface: If True, load from Hugging Face datasets hub
        
    Returns:
        dataloader: DataLoader for the dataset
        dataset: The dataset object
    """
    print(f"Loading dataset from: {dataset_path}")
    
    if from_huggingface:
        if not DATASETS_AVAILABLE:
            raise ImportError(
                "datasets library is required for Hugging Face datasets. "
                "Install with: pip install datasets"
            )
        
        # Load from Hugging Face
        print(f"Loading dataset from Hugging Face: {dataset_path}")
        try:
            hf_dataset = load_dataset(dataset_path, split="train")
            print(f"Hugging Face dataset loaded. Number of samples: {len(hf_dataset)}")
            
            # Convert Hugging Face dataset to a format compatible with our training
            # We'll create a wrapper that converts the format
            dataset = HuggingFaceDatasetWrapper(hf_dataset)
            print(f"Dataset wrapper created. Number of samples: {len(dataset)}")
        except Exception as e:
            raise ValueError(f"Failed to load Hugging Face dataset: {e}")
    else:
        # Load from local path
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
        
        # Load lerobot dataset
        try:
            dataset = LeRobotDataset(dataset_path)
            print(f"Dataset loaded successfully. Number of samples: {len(dataset)}")
        except Exception as e:
            raise ValueError(f"Failed to load dataset: {e}")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=None,  # Use default collate function
    )
    
    return dataloader, dataset


class HuggingFaceDatasetWrapper:
    """
    Wrapper to convert Hugging Face dataset format to format expected by training script.
    Handles the so101 arm dataset structure with observation.images.front, action, etc.
    """
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset
    
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        
        # Extract data according to the dataset structure
        # observation.images.front -> image
        # action -> action
        # observation.state -> state (optional)
        
        result = {}
        
        # Handle images - the dataset has observation.images.front
        if "observation.images.front" in sample:
            image = sample["observation.images.front"]
            # If it's a video frame, extract first frame or handle appropriately
            if isinstance(image, list) and len(image) > 0:
                image = image[0]  # Take first frame if it's a list
            result["image"] = image
            result["observation.images.front"] = image  # Keep original key too
        
        # Handle actions
        if "action" in sample:
            action = sample["action"]
            # Convert to numpy array if it's a list
            if isinstance(action, list):
                action = np.array(action, dtype=np.float32)
            result["action"] = action
        
        # Handle state (optional, for reference)
        if "observation.state" in sample:
            state = sample["observation.state"]
            if isinstance(state, list):
                state = np.array(state, dtype=np.float32)
            result["observation.state"] = state
        
        # Keep other fields
        for key in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
            if key in sample:
                result[key] = sample[key]
        
        return result


def train_step(model, processor, batch, device, optimizer=None, training=True):
    """
    Perform a single training step.
    
    Args:
        model: The model to train
        processor: The processor for tokenization
        batch: Batch of data
        device: Device to run on
        optimizer: Optimizer (None for eval)
        training: Whether in training mode
        
    Returns:
        loss: Training loss
    """
    model.train() if training else model.eval()
    
    # Extract images, actions, and text instructions from batch
    # Handle different possible keys in lerobot dataset
    images = None
    if "image" in batch:
        images = batch["image"]
    elif "frames" in batch:
        images = batch["frames"]
    elif "observation.image" in batch:
        images = batch["observation.image"]
    elif "observation.images.front" in batch:
        # Handle so101 arm dataset structure
        images = batch["observation.images.front"]
    
    actions = batch.get("action", batch.get("actions", None))
    instructions = batch.get("instruction", batch.get("text", batch.get("task_description", None)))
    
    if images is None:
        available_keys = list(batch.keys())
        raise ValueError(
            f"Batch must contain 'image', 'frames', 'observation.image', or 'observation.images.front' key. "
            f"Available keys: {available_keys}"
        )
    
    # Handle single image vs batch of images
    if isinstance(images, list):
        # Convert list of images to tensor if needed
        pass
    elif isinstance(images, torch.Tensor):
        # Already a tensor
        pass
    
    # Prepare inputs
    inputs = processor(
        images=images,
        text=instructions if instructions is not None else [""] * len(images),
        return_tensors="pt",
        padding=True,
    )
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Prepare action targets (convert actions to text format for VLA model)
    # smolvla generates text, so we format actions as text targets
    if actions is not None:
        # Convert actions to text format that smolvla expects
        # Format: "action: [a1, a2, a3, ...]" for so101 arm
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        
        action_texts = []
        for action in actions:
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            # Format action as text for the model to generate
            action_str = ", ".join([f"{a:.4f}" for a in action.flatten()])
            action_texts.append(f"action: [{action_str}]")
        
        # Create full prompts: instruction + action target
        if instructions is None:
            instructions = [""] * len(images)
        
        # Combine instruction and action target
        full_texts = [
            f"{inst}\n{action}" if inst else f"Execute action: {action}"
            for inst, action in zip(instructions, action_texts)
        ]
        
        # Re-process with action targets
        inputs = processor(
            images=images,
            text=full_texts,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
    else:
        # If no actions, just use instructions
        if instructions is None:
            instructions = [""] * len(images)
        inputs = processor(
            images=images,
            text=instructions,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Forward pass
    with torch.set_grad_enabled(training):
        outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
    
    # Backward pass
    if training and optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    return loss.item()


def train(
    dataset_path,
    output_dir="./checkpoints",
    model_name="HuggingFaceTB/smolvla-instruct",
    batch_size=4,
    num_epochs=10,
    learning_rate=1e-5,
    num_workers=4,
    save_steps=500,
    eval_steps=250,
    use_wandb=True,
    wandb_project="smolvla-so101-finetune",
    from_huggingface=False,
):
    """
    Main training function.
    
    Args:
        dataset_path: Path to the lerobot dataset
        output_dir: Directory to save checkpoints
        model_name: HuggingFace model identifier
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        num_workers: Number of data loading workers
        save_steps: Save checkpoint every N steps
        eval_steps: Evaluate every N steps
        use_wandb: Whether to use wandb for logging
        wandb_project: Wandb project name
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(project=wandb_project, config={
            "model_name": model_name,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "dataset_path": dataset_path,
            "robot": "so101_arm",
        })
    
    # Setup model and processor
    model, processor = setup_model_and_processor(model_name)
    model = model.to(device)
    
    # Prepare dataset
    dataloader, dataset = prepare_dataset(dataset_path, batch_size, num_workers, from_huggingface=from_huggingface)
    
    # Setup PyTorch optimizer (NOT lerobot's trainer)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )
    
    # Setup PyTorch learning rate scheduler
    num_training_steps = len(dataloader) * num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_training_steps,
    )
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # PyTorch training loop (manual, not using lerobot's trainer)
    global_step = 0
    print(f"Starting training for {num_epochs} epochs...")
    print("Using PyTorch training loop (not lerobot trainer)")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        
        for batch in progress_bar:
            # Training step
            loss = train_step(
                model, processor, batch, device,
                optimizer=optimizer, training=True
            )
            
            epoch_loss += loss
            num_batches += 1
            global_step += 1
            
            # Update learning rate
            scheduler.step()
            
            # Logging
            if use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "train/loss": loss,
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/epoch": epoch,
                    "train/step": global_step,
                })
            
            progress_bar.set_postfix({"loss": f"{loss:.4f}"})
            
            # Save checkpoint
            if global_step % save_steps == 0:
                checkpoint_dir = output_path / f"checkpoint-{global_step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                model.save_pretrained(checkpoint_dir)
                processor.save_pretrained(checkpoint_dir)
                
                print(f"\nSaved checkpoint at step {global_step} to {checkpoint_dir}")
        
        # Epoch summary
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")
        
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "train/epoch_loss": avg_loss,
                "train/epoch": epoch + 1,
            })
    
    # Save final model
    final_dir = output_path / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"\nSaved final model to {final_dir}")
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    print("Training completed!")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune smolvla model for so101 arm robot imitation learning"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the lerobot dataset directory or Hugging Face dataset identifier (e.g., 'HenryZhang/Group_11_dataset_1763073574.9315236')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints (default: ./checkpoints)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="HuggingFaceTB/smolvla-instruct",
        help="HuggingFace model identifier (default: HuggingFaceTB/smolvla-instruct)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training (default: 4)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate (default: 1e-5)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps (default: 500)",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=250,
        help="Evaluate every N steps (default: 250)",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable wandb logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="smolvla-so101-finetune",
        help="Wandb project name (default: smolvla-so101-finetune)",
    )
    parser.add_argument(
        "--from_huggingface",
        action="store_true",
        help="Load dataset from Hugging Face datasets hub instead of local path",
    )
    
    args = parser.parse_args()
    
    train(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        from_huggingface=args.from_huggingface,
    )


if __name__ == "__main__":
    main()

