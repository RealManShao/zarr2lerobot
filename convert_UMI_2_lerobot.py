"""
Convert a UMI-style dataset to LeRobot v3.0 format.

Usage:
    python convert_zarr_to_lerobot.py \
        --datasetPath /path/to/dataset/folder \
        --repo_id your_hf_username/pressbutton1 \
        --robot_type ARX_L5 \
        [--output_root /optional/output/root] \
        [--fps 30] \
        [--push_to_hub] \
        [--push_private]

Notes:
- Requires: lerobot (the package that contains LeRobotDataset),
            zarr, numpy, tyro (for CLI).
- This script expects the Zarr layout shown in your notes:
    root['data'][...]: arrays concatenated across episodes
    root['meta']['episode_ends']: 1D int array of cumulative end indices
- By default the conversion maps:
    - actions <- data['action'] (7,)
    - state   <- concat(data['joint_pos'] (6,), data['gripper_pos'] (1,)) -> (7,)
    - cartesian_position <- data['eef_pose'] (6,)
    - joint_torque <- data['joint_torque'] (6,)
    - gripper_torque <- data['gripper_torque'] (1,)
    - timestamp <- data['timestamp']
- Video support is built-in: videos are expected in subdirectories named after episode index
  with filenames like "{episode_index}/0.mp4" and "{episode_index}/1.mp4" for wrist and exterior views.

Example:
    python convert_zarr_to_lerobot.py --datasetPath ./pressButton1 --repo_id "me/pressbutton1" --robot_type "ARX_L5"
"""

from pathlib import Path
import sys
import logging

import numpy as np
import zarr
import tyro

from lerobot.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME

# Optional for reading videos
try:
    import imageio
except Exception:
    imageio = None


def convert(
    datasetPath: str,
    repo_id: str,
    *,
    output_root: str | None = None,
    robot_type: str = "ARX_L5",
    fps: int = 30,
    push_to_hub: bool = False,
    push_private: bool = False,
):
    zarr_path = Path(datasetPath / replay_buffer.zarr)
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr file or directory not found: {zarr_path}")

    videos_dir = Path(datasetPath / videos)

    out_root = Path(output_root) if output_root is not None else None

    # Open Zarr
    root = zarr.open(str(zarr_path), mode="r")
    data = root["data"]
    meta = root["meta"]
    episode_ends = meta["episode_ends"][:]
    n_episodes = len(episode_ends)
    logging.info(f"Found {n_episodes} episodes (episode_ends: {episode_ends[:10]}...)")

    # Prepare LeRobot features
    # Basic mapping: actions (7,), state (joint_pos(6,) + gripper_pos(1,) -> 7,)
    UMI_features = {
        "observation.state.cartesian_position": {
            "dtype": "float32",
            "shape": (6,),
            "names": {
                "axes": ["x", "y", "z", "roll", "pitch", "yaw"],
            },
        },
        "observation.state.joint_position": {
            "dtype": "float32",
            "shape": (6,),
            "names": {
                "axes": [
                    "joint_0",
                    "joint_1",
                    "joint_2",
                    "joint_3",
                    "joint_4",
                    "joint_5",
                ],
            },
        },
        "observation.state.joint_torque": {
            "dtype": "float32",
            "shape": (6,),
            "names": {
                "axes": [
                    "joint_0",
                    "joint_1",
                    "joint_2",
                    "joint_3",
                    "joint_4",
                    "joint_5",
                ],
            },
        },
        "observation.state.gripper_position": {
            "dtype": "float32",
            "shape": (1,),
            "names": {
                "axes": ["gripper"],
            },
        },
        "observation.state.gripper_torque": {
            "dtype": "float32",
            "shape": (1,),
            "names": {
                "axes": ["gripper"],
            },
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": {
                "axes": [
                    "joint_0",
                    "joint_1",
                    "joint_2",
                    "joint_3",
                    "joint_4",
                    "joint_5",
                    "gripper",
                ],
            },
        },
        "observation.images.wrist": {
            "dtype": "float32",
            "shape": (720, 1280, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.exterior": {
            "dtype": "float32",
            "shape": (720, 1280, 3),
            "names": ["height", "width", "channels"],
        },
        # Add this new feature to follow LeRobot standard of using joint position + gripper
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": {
                "axes": [
                    "joint_0",
                    "joint_1",
                    "joint_2",
                    "joint_3",
                    "joint_4",
                    "joint_5",
                    "gripper",
                ],
            },
        },
    }

    # Create LeRobot dataset
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=robot_type,
        fps=fps,
        features=UMI_features,
        root=out_root,
    )

    # Build episode ranges (start_idx inclusive, end_idx exclusive)
    starts = [[0] + episode_ends[:-1]].tolist()
    ends = episode_ends.tolist()
    episode_idx = 0

    for episode_idx in range(len(starts)):
        ep_len = ends[episode_idx] - starts[episode_idx]
        logging.info(
            f"Processing episode {episode_idx}: frames {starts[episode_idx]} to {ends[episode_idx]-1} (len={ep_len})"
        )

        # Build frames
        for relative_frame in range(0, ep_len):
            frame_idx = starts[episode_idx] + relative_frame
            # actions
            actions = np.asarray(data["action"][frame_idx], dtype=np.float32)
            # joint and gripper positions
            joint_pos = np.asarray(data["joint_pos"][frame_idx][0:6], dtype=np.float32)
            joint_torque = np.asarray(
                data["joint_torque"][frame_idx][0:6], dtype=np.float32
            )
            gripper_pos = np.asarray(data["gripper_pos"][frame_idx], dtype=np.float32)
            gripper_torque = np.asarray(
                data["gripper_torque"][frame_idx], dtype=np.float32
            )
            eef_pose = data["eef_pose"][frame_idx]
            timestamp = data["timestamp"][frame_idx]

            frame = {
                "actions": actions,
                "observation.state.cartesian_position": eef_pose,
                "observation.state.joint_position": joint_pos,
                "observation.state.joint_torque": joint_torque,
                "observation.state.gripper_position": gripper_pos,
                "observation.state.gripper_torque": gripper_torque,
            }
            frame["observation.state"] = np.concatenate(
                [
                    frame["observation.state.joint_position"],
                    frame["observation.state.gripper_position"],
                ]
            )

            # Add video data to the first frame of each episode only, as video data is typically
            # associated with the episode rather than individual frames
            if relative_frame == 0:
                # For video observations, we add the video data to the first frame
                # LeRobot likely handles the video processing internally
                # Look for the episode video file (e.g., episode_0.mp4)
                wrist_video_path = videos_dir.glob(f"{episode_idx}/0.mp4")
                exterior_video_path = videos_dir.glob(f"{episode_idx}/1.mp4")

                frame["observation.images.wrist"] = str(wrist_video_path)
                frame["observation.images.exterior"] = str(exterior_video_path)

            # Add frame to dataset
            dataset.add_frame(frame)
            relative_frame += 1

        # save episode (writes parquet metadata and triggers video encoding if needed)
        dataset.save_episode()
        episode_idx += 1

    # Finalize dataset (close writers and flush metadata)
    dataset.finalize()

    # optionally push to hub
    if push_to_hub:
        logging.info("Pushing dataset to the Hugging Face Hub...")
        dataset.push_to_hub(
            tags=[repo_id.split("/")[-1], "converted", "uim"], private=push_private
        )

    logging.info("Conversion finished. Dataset root: %s", (dataset.root))

    if __name__ == "__main__":
        tyro.extras.set_accent_color("green")
        tyro.cli(convert)
