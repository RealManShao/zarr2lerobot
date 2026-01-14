from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Load your local dataset
dataset = LeRobotDataset("Xihe666/ARX_L5_WipeBoard", root="./output_merged")

# Push to Hugging Face Hub
dataset.push_to_hub(
    tags=["ARX L5", "wipe_board"],
    private=False,
)