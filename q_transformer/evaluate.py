import gymnasium as gym
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper

from PIL import Image

import numpy as np

import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from q_transformer import QRoboticTransformer

ACTION_DIM = 7
DELTA_BOUNDS = np.array(
    [[-1.0 for i in range(ACTION_DIM)], [1.0 for i in range(ACTION_DIM)]]
)
NUM_BINS = 256
NUM_RUNS = 50
IMG_H = IMG_W = 112

CKPT_FILE = "./checkpoints/checkpoint-10.pt"

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

def from_bins(action):
    action = DELTA_BOUNDS[0] + action.numpy() / NUM_BINS * (
        DELTA_BOUNDS[1] - DELTA_BOUNDS[0]
    )
    return action



if __name__ == "__main__":

    model = QRoboticTransformer(
        vit = dict(
            num_classes = 1000,
            dim_conv_stem = 64,
            dim = 64,
            dim_head = 64,
            depth = (2, 2, 5, 2),
            window_size = 7,
            mbconv_expansion_rate = 4,
            mbconv_shrinkage_rate = 0.25,
            dropout = 0.1,
            conv_stem_downsample = False, # we are using 112x112 which is rather low
        ),
        num_actions = 7,
        action_bins = 256,
        depth = 1,
        heads = 8,
        dim_head = 64,
        cond_drop_prob = 0.2,
        dueling = True
    )

    model.load_state_dict(torch.load(CKPT_FILE)["model"])

    model.eval()
    text_embed = model.embed_texts(["push cube"])

    env = gym.make(
        "PushCube-v1",
        obs_mode="rgbd",
        control_mode="pd_ee_delta_pose",
        num_envs=1,
        render_mode="human",
    )
    env = CPUGymWrapper(env)

    instructions = ['push cube']
    episode_rewards = []
    transform = _transform(IMG_W)

    for _ in range(NUM_RUNS):
        (state, _), done, terminated = env.reset(), False, False
        episode_reward = 0.0
        while not (done or terminated):
            frame = transform(
                Image.fromarray(
                    state["sensor_data"]["base_camera"]["rgb"]
                )
            ).unsqueeze(dim=1).unsqueeze(dim=0)
            action = from_bins(model.get_optimal_actions(frame, instructions)[0])
            state, reward, done, terminated, _ = env.step(action)
            episode_reward += reward
            env.render()
        print("episode reward = ", episode_reward)
        episode_rewards.append(episode_reward)
    
    print("total average eval reward = ", sum(episode_rewards) / len(episode_rewards))

