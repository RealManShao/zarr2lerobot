import numpy as np

from lerobot.datasets.dataset_tools import (
    add_features,
    delete_episodes,
    merge_datasets,
    modify_features,
    remove_feature,
    split_dataset,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def main():
    dataset = [
        LeRobotDataset("my_local_dataset1", root="./output"),
        LeRobotDataset("my_local_dataset2", root="./output20"),
        LeRobotDataset("my_local_dataset3", root="./output30"),
    ]
    merge_datasets(dataset, output_repo_id = "Xihe666/ARX_L5_WipeBoard", output_dir="./output_merged")

if __name__ == "__main__":
    main()