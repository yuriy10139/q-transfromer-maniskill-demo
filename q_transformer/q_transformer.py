from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
from torch.utils.data import Dataset, DataLoader

from einops import rearrange
from einops.layers.torch import Rearrange

from beartype import beartype
from beartype.typing import Optional, Union, List, Tuple

from q_transformer.robotic_transformer import QRoboticTransformer

from q_transformer.optimizer import get_adam_optimizer

from accelerate import Accelerator

from ema_pytorch import EMA

# helpers

def exists(val):
    return val is not None

def is_divisible(num, den):
    return (num % den) == 0

def cycle(dl):
    while True:
        for batch in dl:
            yield batch

# Q learning on robotic transformer

class QLearner(Module):

    @beartype
    def __init__(
        self,
        model: Union[QRoboticTransformer, Module],
        *,
        dataset: Dataset,
        batch_size: int,
        num_train_steps: int,
        learning_rate: float,
        weight_decay: float = 0.,
        accelerator: Optional[Accelerator] = None,
        accelerator_kwargs: dict = dict(),
        dataloader_kwargs: dict = dict(
            shuffle = True
        ),
        ema_kwargs: dict = dict(
            beta = 0.999,
            update_after_step = 10,
            update_every = 5
        ),
        optimizer_kwargs: dict = dict(),
        checkpoint_folder = './checkpoints',
        checkpoint_every = 1000,
    ):
        super().__init__()
        assert model.num_actions == 1

        # online (evaluated) Q model

        self.model = model

        # ema (target) Q model

        self.ema_model = EMA(
            model,
            include_online_model = False,
            **ema_kwargs
        )

        self.optimizer = get_adam_optimizer(
            model.parameters(),
            lr = learning_rate,
            wd = weight_decay,
            **optimizer_kwargs
        )

        if not exists(accelerator):
            accelerator = Accelerator(**accelerator_kwargs)

        self.accelerator = accelerator

        self.dataloader = DataLoader(
            dataset,
            batch_size = batch_size,
            **dataloader_kwargs
        )

        # prepare

        (
            self.model,
            self.ema_model,
            self.optimizer,
            self.dataloader
        ) = self.accelerator.prepare(
            self.model,
            self.ema_model,
            self.optimizer,
            self.dataloader
        )

        # checkpointing related

        self.checkpoint_every = checkpoint_every
        self.checkpoint_folder = Path(checkpoint_folder)

        self.checkpoint_folder.mkdir(exist_ok = True, parents = True)
        assert self.checkpoint_folder.is_dir()

        # training step related

        self.num_train_steps = num_train_steps
        self.register_buffer('step', torch.tensor(0))

    def save(
        self,
        checkpoint_num = None,
        overwrite = True
    ):
        name = 'checkpoint'
        if exists(checkpoint_num):
            name += f'-{checkpoint_num}'

        path = self.checkpoint_folder / (name + '.pt')

        assert overwrite or not path.exists()

        pkg = dict(
            model = self.unwrap(self.model).state_dict(),
            ema_model = self.unwrap(self.ema_model).state_dict(),
            optimizer = self.optimizer.state_dict()
        )

        torch.save(pkg, str(path))

    def load(self, path):
        path = Path(path)
        assert exists(path)

        pkg = torch.load(str(path))

        self.unwrap(self.model).load_state_dict(pkg['model'])
        self.unwrap(self.ema_model).load_state_dict(pkg['ema_model'])

        self.optimizer.load_state_dict(pkg['optimizer'])

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def unwrap(self, module):
        return self.accelerator.unwrap_model(module)

    def print(self, msg):
        return self.accelerator.print(msg)

    def wait(self):
        return self.accelerator.wait_for_everyone()

    def forward(self):
        step = self.step.item()

        while step < self.num_train_steps:

            step += 1
            self.print(step)
            self.step.add_(1)

            # whether to checkpoint or not

            self.wait()

            if self.is_main and is_divisible(step, self.checkpoint_every):
                checkpoint_num = step // self.checkpoint_every
                self.save(checkpoint_num)

            self.wait()

        self.print('training complete')
