"""
Convert a UMI-style dataset to LeRobot v3.0 format.

Usage:
    python convert_UMI_2_lerobot.py \
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
  When video frame count > episode frame count, linear mapping is used to find corresponding frames.
  If exact frame mapping fails, nearby frames (n-1, n-2, n-3) are tried as fallback.

Example:
    python convert_UMI_2_lerobot.py --datasetPath /home/phi/Documents/zarr/ARX_dataset_wipe/wipe30 --repo_id "Xihe666/ARX_L5_WipeBoard" --output_root ./output30
"""

from pathlib import Path
import sys
import logging

import numpy as np
import zarr
import tyro
import cv2
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME


def get_video_frame_count(video_path: str) -> int:
    """
    Get the total number of frames in a video.

    Args:
        video_path: Path to the video file

    Returns:
        Total number of frames in the video
    """
    if not Path(video_path).exists():
        return 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames


def extract_frame_from_video(video_path: str, frame_idx: int) -> np.ndarray:
    """
    Extract a single frame from video at specified frame index.

    Args:
        video_path: Path to the video file
        frame_idx: Frame index to extract

    Returns:
        Numpy array with shape (H, W, 3) and dtype uint8, or None if frame not found
    """
    if not Path(video_path).exists():
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_idx >= total_frames:
            return None

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame_rgb
        else:
            return None
    finally:
        cap.release()


def map_episode_frame_to_video_frame(episode_frame_idx: int, episode_length: int, video_frame_count: int) -> int:
    """
    Map episode frame index to corresponding video frame index using linear mapping.

    Args:
        episode_frame_idx: Index of frame in episode (0 to episode_length-1)
        episode_length: Total number of frames in episode
        video_frame_count: Total number of frames in video

    Returns:
        Corresponding video frame index
    """
    if episode_length == 0 or video_frame_count == 0:
        return 0

    # Linear mapping: episode_frame_idx / episode_length ≈ video_frame_idx / video_frame_count
    # Solve for video_frame_idx: video_frame_idx = episode_frame_idx * video_frame_count / episode_length
    video_frame_idx = int(episode_frame_idx * video_frame_count / episode_length)

    # Ensure we don't exceed video bounds
    return min(video_frame_idx, video_frame_count - 1)


def convert(
    datasetPath: str,
    repo_id: str,
    *,
    output_root: str | None = None,
    robot_type: str = "ARX_L5",
    fps: int = 10,
    push_to_hub: bool = False,
    push_private: bool = False,
):
    zarr_path = Path(datasetPath) / "replay_buffer.zarr"
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr file or directory not found: {zarr_path}")

    videos_dir = Path(datasetPath) / "videos"

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
            "language_instruction": {
                "dtype": "string",
                "shape": (1,),
                "names": None,
            },
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
            "dtype": "video",
            "shape": (720, 1280, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.exterior": {
            "dtype": "video",
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
    starts = np.insert(episode_ends[:-1], 0, 0)
    ends = episode_ends
    episode_idx = 0

    for episode_idx in range(0, n_episodes):
        ep_len = ends[episode_idx] - starts[episode_idx]
        logging.info(
            f"Processing episode {episode_idx}: frames {starts[episode_idx]} to {ends[episode_idx]-1} (len={ep_len})"
        )

        # Get video information for this episode
        wrist_video_path = videos_dir / f"{episode_idx}/0.mp4"
        exterior_video_path = videos_dir / f"{episode_idx}/1.mp4"

        wrist_frame_count = get_video_frame_count(str(wrist_video_path))
        exterior_frame_count = get_video_frame_count(str(exterior_video_path))

        if wrist_video_path.exists() and wrist_frame_count > 0:
            logging.info(f"Wrist video found: {wrist_frame_count} frames")
        else:
            logging.warning(f"Wrist video not found or empty: {wrist_video_path}")

        if exterior_video_path.exists() and exterior_frame_count > 0:
            logging.info(f"Exterior video found: {exterior_frame_count} frames")
        else:
            logging.warning(f"Exterior video not found or empty: {exterior_video_path}")

        # Log frame count comparison
        if wrist_frame_count > 0:
            logging.info(f"Episode {episode_idx}: {ep_len} episode frames, {wrist_frame_count} wrist video frames")
        if exterior_frame_count > 0:
            logging.info(f"Episode {episode_idx}: {ep_len} episode frames, {exterior_frame_count} exterior video frames")

        # Build frames
        for relative_frame in range(0, ep_len):
            frame_idx = starts[episode_idx] + relative_frame

            # Build frame following DROID port structure
            frame = {
                # Actions
                "action": np.asarray(data["action"][frame_idx], dtype=np.float32),

                # State observations
                "observation.state.cartesian_position": np.asarray(data["eef_pose"][frame_idx], dtype=np.float32),
                "observation.state.joint_position": np.asarray(data["joint_pos"][frame_idx][0:6], dtype=np.float32),
                "observation.state.joint_torque": np.asarray(data["joint_torque"][frame_idx][0:6], dtype=np.float32),
                "observation.state.gripper_position": np.asarray(data["gripper_pos"][frame_idx], dtype=np.float32),
                "observation.state.gripper_torque": np.asarray(data["gripper_torque"][frame_idx], dtype=np.float32),
            }

            # Add combined state (joint + gripper positions) following LeRobot standard
            frame["observation.state"] = np.concatenate([
                frame["observation.state.joint_position"],
                frame["observation.state.gripper_position"]
            ])

            # Add video frames using linear mapping (only when video frames > episode frames)
            if wrist_frame_count > ep_len:
                # Map episode frame to video frame using linear mapping
                video_frame_idx = map_episode_frame_to_video_frame(relative_frame, ep_len, wrist_frame_count)
                wrist_frame = extract_frame_from_video(str(wrist_video_path), video_frame_idx)

                # If exact frame not found, try nearby frames (n-1, n-2, n-3)
                if wrist_frame is None:
                    for offset in [1, 2, 3]:
                        if video_frame_idx >= offset:
                            wrist_frame = extract_frame_from_video(str(wrist_video_path), video_frame_idx - offset)
                            if wrist_frame is not None:
                                break

                # If still no frame found, use black placeholder
                if wrist_frame is None:
                    wrist_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

                frame["observation.images.wrist"] = wrist_frame
            else:
                # Video has fewer or equal frames than episode - use black placeholder
                frame["observation.images.wrist"] = np.zeros((720, 1280, 3), dtype=np.uint8)

            if exterior_frame_count > ep_len:
                # Map episode frame to video frame using linear mapping
                video_frame_idx = map_episode_frame_to_video_frame(relative_frame, ep_len, exterior_frame_count)
                exterior_frame = extract_frame_from_video(str(exterior_video_path), video_frame_idx)

                # If exact frame not found, try nearby frames (n-1, n-2, n-3)
                if exterior_frame is None:
                    for offset in [1, 2, 3]:
                        if video_frame_idx >= offset:
                            exterior_frame = extract_frame_from_video(str(exterior_video_path), video_frame_idx - offset)
                            if exterior_frame is not None:
                                break

                # If still no frame found, use black placeholder
                if exterior_frame is None:
                    exterior_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

                frame["observation.images.exterior"] = exterior_frame
            else:
                # Video has fewer or equal frames than episode - use black placeholder
                frame["observation.images.exterior"] = np.zeros((720, 1280, 3), dtype=np.uint8)

            # language_instruction is also stored as "task" to follow LeRobot standard
            frame["language_instruction"] = "wipe_board"
            frame["task"] = frame["language_instruction"]


            # Add frame to dataset
            dataset.add_frame(frame)

        # save episode (writes parquet metadata and triggers video encoding if needed)
        dataset.save_episode()

    # Finalize dataset (close writers and flush metadata)
    dataset.finalize()

    # optionally push to hub
    if push_to_hub:
        logging.info("Pushing dataset to the Hugging Face Hub...")
        dataset.push_to_hub(
            tags=[repo_id.split("/")[-1], "ARX L5"], private=push_private
        )

    logging.info("Conversion finished. Dataset root: %s", (dataset.root))


if __name__ == "__main__":
    tyro.extras.set_accent_color("green")
    tyro.cli(convert)
