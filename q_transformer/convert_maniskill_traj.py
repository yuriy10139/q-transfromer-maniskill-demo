from pathlib import Path
import h5py
from tqdm import tqdm

from PIL import Image

import numpy as np
from numpy.lib.format import open_memmap

import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from q_transformer import QRoboticTransformer
from mani_skill.utils.io_utils import load_json

# PushCube-v1 action space is Box(-1.0, 1.0, (7,), float32)
ACTION_DIM = 7
DELTA_BOUNDS = np.array(
    [[-1.0 for i in range(ACTION_DIM)], [1.0 for i in range(ACTION_DIM)]]
)

TEXT_EMBEDS_FILENAME = "text_embeds.memmap.npy"
STATES_FILENAME = "states.memmap.npy"
ACTIONS_FILENAME = "actions.memmap.npy"
REWARDS_FILENAME = "rewards.memmap.npy"
DONES_FILENAME = "dones.memmap.npy"

DEFAULT_REPLAY_MEMORIES_FOLDER = "./replay_memories_data"
DATASET_FILE = "./demos/PushCube-v1/motionplanning/trajectory.rgbd.pd_ee_delta_pose.h5"

NUM_CAM = 1
IMG_H = IMG_W = 112


def make_bins(action, bins=256):
    bin_size = (DELTA_BOUNDS[1] - DELTA_BOUNDS[0]) / bins
    action = np.clip(
        ((action - DELTA_BOUNDS[0]) / bin_size).astype(int),
        [0 for i in range(ACTION_DIM)],
        [bins - 1 for i in range(ACTION_DIM)],
    )
    return action


def flush_all():
    states.flush()
    actions.flush()
    rewards.flush()
    dones.flush()
    text_embeds.flush()
    return


def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out


def _transform(n_px):
    return Compose(
        [
            # Resize(n_px, interpolation=Image.Resampling.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


if __name__ == "__main__":
    model = QRoboticTransformer(
        vit=dict(
            num_classes=1000,
            dim_conv_stem=64,
            dim=64,
            dim_head=64,
            depth=(2, 2, 5, 2),
            window_size=7,
            mbconv_expansion_rate=4,
            mbconv_shrinkage_rate=0.25,
            dropout=0.1,
        ),
        num_actions=8,
        action_bins=256,
        depth=1,
        heads=8,
        dim_head=64,
        cond_drop_prob=0.2,
        dueling=True,
    )

    model.eval()
    text_embed = model.embed_texts(["push cube"])

    data = h5py.File(DATASET_FILE, "r")
    json_data = load_json(DATASET_FILE.replace(".h5", ".json"))
    episodes = json_data["episodes"]
    env_info = json_data["env_info"]
    env_id = env_info["env_id"]
    env_kwargs = env_info["env_kwargs"]
    load_count = len(episodes)

    # max_traj_len = 0
    # for ep in episodes:
    #      if ep['elapsed_steps'] > max_traj_len:
    #           max_traj_len = ep['elapsed_steps']
    #           print(max_traj_len, ep['episode_id'])

    max_traj_len = 75  # find 'acceptable' vaule of this variable using these commented section

    mem_path = Path(DEFAULT_REPLAY_MEMORIES_FOLDER)
    mem_path.mkdir(exist_ok = True, parents = True)
    assert mem_path.is_dir()

    states_path = mem_path / STATES_FILENAME
    actions_path = mem_path / ACTIONS_FILENAME
    rewards_path = mem_path / REWARDS_FILENAME
    dones_path = mem_path / DONES_FILENAME
    text_embeds_path = mem_path / TEXT_EMBEDS_FILENAME

    states      = open_memmap(str(states_path), dtype = 'float16', mode = 'w+', shape = (load_count, max_traj_len, 3, NUM_CAM, IMG_W, IMG_H))
    actions     = open_memmap(str(actions_path), dtype = 'int', mode = 'w+', shape = (load_count, max_traj_len, ACTION_DIM))
    rewards     = open_memmap(str(rewards_path), dtype = 'float32', mode = 'w+', shape = (load_count, max_traj_len))
    dones       = open_memmap(str(dones_path), dtype = 'bool', mode = 'w+', shape = (load_count, max_traj_len))
    text_embeds = open_memmap(str(text_embeds_path), dtype = 'float32', mode = 'w+', shape = (load_count, max_traj_len, 768))

    transform = _transform(IMG_W)

    for eps_id in tqdm(range(load_count)):
        eps = episodes[eps_id]
        if eps["elapsed_steps"] > max_traj_len:
            continue
        trajectory = data[f"traj_{eps['episode_id']}"]
        trajectory = load_h5_data(trajectory)
        reward = np.concatenate(
            (np.where(trajectory["success"], 1.0, 0.0)[1:], np.array([1.0]))
        )

        rewards[eps_id, : len(reward)] = reward
        dones[eps_id, : len(reward)] = trajectory["success"]
        text_embeds[eps_id, :] = text_embed[0]
        for step in range(len(reward)):
            states[eps_id, step, ::] = transform(
                Image.fromarray(
                    trajectory["obs"]["sensor_data"]["base_camera"]["rgb"][step]
                )
            ).unsqueeze(dim=1)
            actions[eps_id, step, :] = make_bins(trajectory["actions"][step])
        flush_all()
