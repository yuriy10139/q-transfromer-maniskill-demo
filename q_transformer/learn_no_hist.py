import torch

from q_transformer import (
    QRoboticTransformer,
    QLearner,
    ReplayMemoryDataset
)

CKPT_FILE = None

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

if CKPT_FILE:
    model.load_state_dict(torch.load(CKPT_FILE)["model"])

q_learner = QLearner(
    model,
    dataset = ReplayMemoryDataset(),
    num_train_steps = 10000,
    learning_rate = 1e-5,
    batch_size = 12,
    grad_accum_every = 8,
    checkpoint_every = 1000,
)

q_learner()