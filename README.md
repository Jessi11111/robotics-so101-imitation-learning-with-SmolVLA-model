# robotics-so101-imitation-learning-with-SmolVLA-model

1. venv:
python3 -m venv robot
source robot/bin/activate



2.
macOS/Windows (Native PowerShell):
pip install -r requirements.txt

WSL/Linux:
# First, try the standard requirements
pip install -r requirements.txt

# If you get evdev build errors, use this instead:
pip install -r requirements-wsl.txt











2. dataset from huggingface


run command:
python fine_tune.py --dataset_path "HenryZhang/Group_11_dataset_1763073574.9315236" --from_huggingface
