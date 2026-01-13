"""
Convert a Zarr / UMI-style dataset to LeRobot v3.0 format.

Usage:
    python convert_zarr_to_lerobot.py \
        --zarr_path /path/to/replay_buffer.zarr \
        --repo_id your_hf_username/pressbutton1 \
        --output_root /optional/output/root \
        [--videos_dir /path/to/videos] \
        [--fps 60] \
        [--use_videos] \
        [--push_to_hub]

Notes:
- Requires: lerobot (the package that contains LeRobotDataset),
            zarr, numpy, imageio (optional, for reading mp4),
            tyro (for CLI).
- This script expects the Zarr layout shown in your notes:
    root['data'][...]: arrays concatenated across episodes
    root['meta']['episode_ends']: 1D int array of cumulative end indices
- By default the conversion maps:
    - actions <- data['action'] (7,)
    - state   <- concat(data['joint_pos'] (7,), data['gripper_pos'] (1,)) -> (8,)
    - timestamp <- data['timestamp']
    - task <- "stage_{stage}" (per-frame; episode-level tasks inferred from unique values)
- If --use_videos is set and a videos_dir is provided, the script will try to read each episode's video
  (mp4 or image sequence) and add images under feature key "observation" (dtype "image").
  The video frame shape will be detected from the first frame encountered.

Example:
    python convert_zarr_to_lerobot.py --zarr_path ./replay_buffer.zarr --repo_id "me/pressbutton1" --videos_dir ./videos --use_videos
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


def _read_video_frames_for_episode(videos_dir: Path, ep_idx: int):
    """
    Try several strategies to read frames for episode `ep_idx`:
    - videos_dir/<ep_idx>/*.png or *.jpg  (image sequence)
    - videos_dir/<ep_idx>.mp4 or videos_dir/<ep_idx>/*.mp4 (single file)
    Returns a list of HxWxC uint8 numpy arrays.
    """
    ep_dir = videos_dir / str(ep_idx)
    frames = []

    # 1) image sequence inside a folder
    if ep_dir.is_dir():
        imgs = sorted(list(ep_dir.glob("*.png")) + list(ep_dir.glob("*.jpg")) + list(ep_dir.glob("*.jpeg")))
        if len(imgs) > 0:
            for p in imgs:
                img = imageio.v2.imread(str(p))
                frames.append(img)
            return frames

    # 2) single mp4 file named by episode
    candidates = list(videos_dir.glob(f"{ep_idx}.*")) + list(videos_dir.glob(f"{ep_idx}/*.mp4")) + list(videos_dir.glob(f"{ep_idx}/*.mov"))
    # fallback: any mp4 in videos_dir root
    if len(candidates) == 0:
        candidates = list(videos_dir.glob("*.mp4")) + list(videos_dir.glob("*.mov"))

    for cand in candidates:
        if cand.suffix.lower() in {".mp4", ".mov", ".mkv"} or cand.is_file():
            try:
                reader = imageio.get_reader(str(cand))
                for frame in reader:
                    frames.append(frame)
                reader.close()
                if len(frames) > 0:
                    return frames
            except Exception:
                continue

    return frames  # may be empty


def convert(
    zarr_path: str,
    repo_id: str,
    *,
    output_root: str | None = None,
    robot_type: str = "panda",
    fps: int = 60,
    use_videos: bool = False,
    videos_dir: str | None = None,
    image_writer_threads: int = 4,
    image_writer_processes: int = 0,
    push_to_hub: bool = False,
    push_private: bool = False,
):
    zarr_path = Path(zarr_path)
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr file or directory not found: {zarr_path}")

    videos_dir = Path(videos_dir) if videos_dir is not None else None
    if use_videos and (videos_dir is None or not videos_dir.exists()):
        raise FileNotFoundError("use_videos=True but videos_dir not found or not provided.")

    if use_videos and imageio is None:
        raise RuntimeError("imageio is required to read videos/images. Install with `pip install imageio`")

    out_root = Path(output_root) if output_root is not None else None

    # Open Zarr
    root = zarr.open(str(zarr_path), mode="r")
    data = root["data"]
    meta = root["meta"]
    episode_ends = meta["episode_ends"][:]
    n_episodes = len(episode_ends)
    logging.info(f"Found {n_episodes} episodes (episode_ends: {episode_ends[:10]}...)")

    # Prepare LeRobot features
    # Basic mapping: actions (7,), state (joint_pos(7,) + gripper_pos(1,) -> 8,)
    features = {
        "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
        "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
    }

    # If we will attach images, we will detect one frame to set shape later and mutate features accordingly.
    include_observation = False
    obs_shape = None

    if use_videos:
        # try to find at least one frame to infer shape
        sample_frames = None
        for ep_idx in range(n_episodes):
            sample_frames = _read_video_frames_for_episode(videos_dir, ep_idx)
            if len(sample_frames) > 0:
                obs_shape = sample_frames[0].shape  # H, W, C
                include_observation = True
                break
        if include_observation:
            features["observation"] = {
                "dtype": "image",
                "shape": tuple(obs_shape),
                "names": ["height", "width", "channel"],
            }
            logging.info(f"Detected observation shape {obs_shape}, will add 'observation' image feature.")
        else:
            logging.warning("use_videos=True but no frames could be read; proceeding without visual features.")
            include_observation = False

    # Create LeRobot dataset
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=robot_type,
        fps=fps,
        features=features,
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
        root=out_root,
    )

    # Build episode ranges (start_idx inclusive, end_idx exclusive)
    starts = [0] + episode_ends[:-1].tolist()
    ends = episode_ends.tolist()

    global_idx = 0
    for ep_idx, (start, end) in enumerate(zip(starts, ends)):
        ep_len = end - start
        logging.info(f"Processing episode {ep_idx}: frames {start}..{end-1} (len={ep_len})")

        # if using videos, attempt to load frames for this episode
        ep_frames = []
        if include_observation:
            ep_frames = _read_video_frames_for_episode(videos_dir, ep_idx)
            if len(ep_frames) == 0:
                logging.warning(f"No visual frames found for episode {ep_idx}; observation will be filled with zeros.")
            else:
                # If video provides fewer frames than ep_len, we will use as many as available and pad/clip later.
                pass

        # Build frames
        for relative_frame in range(ep_len):
            idx = start + relative_frame
            # actions
            actions = np.asarray(data["action"][idx], dtype=np.float32)
            # joint positions and gripper
            joint_pos = np.asarray(data["joint_pos"][idx], dtype=np.float32)
            # some datasets store gripper as shape (1,) or scalar
            gripper = np.asarray(data["gripper_pos"][idx], dtype=np.float32).ravel()
            if gripper.size == 0:
                # fallback to gripper_pos not present
                gripper = np.array([0.0], dtype=np.float32)
            # ensure shape (1,)
            if gripper.ndim == 0:
                gripper = np.array([float(gripper)], dtype=np.float32)
            state = np.concatenate([joint_pos, gripper]).astype(np.float32)

            frame = {"actions": actions, "state": state}
            # timestamp if present
            if "timestamp" in data:
                frame["timestamp"] = float(data["timestamp"][idx])
            else:
                frame["timestamp"] = float(relative_frame) / float(fps)

            # task: use stage -> "stage_<value>"
            if "stage" in data:
                try:
                    stage_val = int(data["stage"][idx])
                    frame["task"] = f"stage_{stage_val}"
                except Exception:
                    frame["task"] = "stage_0"
            else:
                frame["task"] = "task_0"

            # observation image (if available)
            if include_observation:
                if relative_frame < len(ep_frames):
                    img = ep_frames[relative_frame]
                    # ensure uint8 HWC
                    if img.dtype != np.uint8:
                        img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
                    # convert grayscale to RGB
                    if img.ndim == 2:
                        img = np.stack([img] * 3, axis=-1)
                    frame["observation"] = img
                else:
                    # pad with zeros
                    h, w, c = obs_shape
                    frame["observation"] = np.zeros((h, w, c), dtype=np.uint8)

            # Add frame to dataset
            dataset.add_frame(frame)

        # save episode (writes parquet metadata and triggers video encoding if needed)
        dataset.save_episode()
        global_idx += ep_len

    # Finalize dataset (close writers and flush metadata)
    dataset.finalize()

    # optionally push to hub
    if push_to_hub:
        logging.info("Pushing dataset to the Hugging Face Hub...")
        dataset.push_to_hub(tags=[repo_id.split("/")[-1], "converted", "uim"], private=push_private)

    logging.info("Conversion finished. Dataset root: %s", (dataset.root))


if __name__ == "__main__":
    tyro.extras.set_accent_color("green")
    tyro.cli(convert)