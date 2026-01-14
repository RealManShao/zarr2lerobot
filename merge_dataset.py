from lerobot.datasets.dataset_tools import (
    add_features,
    delete_episodes,
    merge_datasets,
    modify_features,
    remove_feature,
    split_dataset,
)

def main():
    dataset_paths = [
        "./output",
        "./output20",
        "./output30",
    ]
    output_path = "./output_merged"
    merge_datasets(dataset_paths, "Xihe666/ARX_L5_WipeBoard", output_path)

if __name__ == "__main__":
    main()