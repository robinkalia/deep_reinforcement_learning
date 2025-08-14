import cv2
import numpy as np

from typing import List, Generator, Tuple

import torch
from torch import nn, optim
from torch.utils.tensorboard.writer import SummaryWriter
import torch.utils as vutils

import gymnasium as gym
from gymnasium import spaces

import ArgumentParser
import random

import time

LATENT_VEC_SIZE = 100

PROCESSING_WIDTH = 128
PROCESSING_HEIGHT = 128

NUM_DISCR_CHANNELS = 128
NUM_GENER_CHANNELS = 128

BATCH_SIZE = 64

IMAGE_MEAN_THRESH = 0.01

NUM_STEPS_WRITE_REPORT = 100
NUM_STEPS_SAVE_IMAGES = 1000

MAX_NUM_ITERS = 1000000

# Input wrapper for the environments whose purpose is to basically return images of original size: 210x160
# and resize them to a target size of 128x128 and return them.
class InputWrapper(gym.ObservationWrapper):
    def __init__(self, *args):
        super(InputWrapper, self).__init__(args)
        curr_space = self.observation_space
        assert isinstance(curr_space, spaces.Box)
        self.observation_space = spaces.Box(
            self.observation(curr_space.low), self.observation(curr_space.high), dtype=np.float32
        )

    def observation(self, obs: gym.core.ObsType) -> gym.core.ObsType:
        resized_im = cv2.resize(obs, (PROCESSING_WIDTH, PROCESSING_HEIGHT))
        # Transform from (w, h, c) to (c, w, h). Move channel column to first dimension.
        resized_im = np.moveaxis(resized_im, 2, 0)
        return resized_im.astype(np.float32)


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        # Converts the input image into a final single floating point number at the end.
        self.discriminator_model = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=NUM_DISCR_CHANNELS, kernel_size=4, stride=2, padding=1)
            nn.ReLU(),
            nn.Conv2d(in_channels=NUM_DISCR_CHANNELS, out_channels=NUM_DISCR_CHANNELS*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(NUM_DISCR_CHANNELS*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=NUM_DISCR_CHANNELS*2, out_channels=NUM_DISCR_CHANNELS*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(NUM_DISCR_CHANNELS*4),
            nn.ReLU(),
            nn.Conv2d(in_channels=NUM_DISCR_CHANNELS*4, out_channels=NUM_DISCR_CHANNELS*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(NUM_DISCR_CHANNELS*8),
            nn.ReLU(),
            nn.Conv2d(in_channels=NUM_DISCR_CHANNELS*8, out_channels=1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        res = self.discriminator_model(x)
        return res.view(-1, 1).squeeze(dim=1)


class Generator(nn.Module):
    def __init__(self, output_shape):
        super(Generator, self).__init__()
        # Converts a given vector to an image of size (3, 64, 64) at the end.
        self.generator_model = nn.Sequential(
            nn.ConvTranspose2d(in_channels=LATENT_VEC_SIZE, out_channels=NUM_GENER_CHANNELS*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(NUM_GENER_CHANNELS * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=NUM_GENER_CHANNELS*8, out_channels=NUM_GENER_CHANNELS*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(NUM_GENER_CHANNELS * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=NUM_GENER_CHANNELS*4, out_channels=NUM_GENER_CHANNELS*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(NUM_GENER_CHANNELS * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=NUM_GENER_CHANNELS*2, out_channels=NUM_GENER_CHANNELS, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(NUM_GENER_CHANNELS),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=NUM_GENER_CHANNELS, out_channels=output_shape[0], kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self,x):
        res = self.generator_model(x)
        return res.view(-1,1).squeeze(1)


def iterate_batches(envs: List[gym.Env], batch_size: int=BATCH_SIZE) -> Generator[tensor.torch, None, None]:
    batches = [env.reset()[0] for env in envs]

    env = random.choice(envs)
    while True:
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)
        if np.mean(obs) > IMAGE_MEAN_THRESH:
            batches.append(obs)

        # There is an implicit assumption here that batch size will in general be greater than total number of environments.
        # Ideally we should have that because each environment corresponds to a different game and so if you have a lot of
        # games it will take a lot of iterations and the results will not be that good from this simple GAN architecture for
        # image synthesis.
        if (len(batches) == BATCH_SIZE):
            batch_images = np.array(batches, dtype=np.float32)
            norm_images = torch.tensor(batch_images * 2.0 / 255.0 - 1.0)
            yield norm_images
            batches.clear()

        if done or trunc:
            env.reset()

def get_elapsed_time(start_time, curr_time) -> Tuple[int, int, int]:
    train_time_secs = curr_time - start_time
    elapsed_time_hrs = int(train_time_secs / 3600.0)
    elapsed_time_mins = int((train_time_secs - 60*elapsed_time_hrs)  / 60.0)
    elapsed_time_secs = int(train_time_secs - 60 * elapsed_time_mins - 3600 * elapsed_time_hrs)

    return elapsed_time_hrs, elapsed_time_mins, elapsed_time_secs


def train(envs: List[gym.Env], writer: SummaryWriter, device: torch.device="cpu"):

    gen_losses = []
    dis_losses = []

    generator_network = Generator(input_shape)
    discriminator_network = Discriminator(output_shape)

    gen_optim = optim.Adam(generator_network.parameters(), lr=0.001, weight_decay=1e-3, betas=(0.9, 0.99))
    disc_optim = optim.Adam(discriminator_network.parameters(), lr=0.001, weight_decay=1e-3, betas=(0.9, 0.99))


    start_time = time.time()

    num_iters = 0
    for batch in iterate_batches(envs):

        disc_optim.zero_grad()


        disc_loss.backward()
        disc_losses.append(disc_loss)
        disc_optim.step()


        gen_loss.zero_grad()


        gen_loss.backward()
        gen_losses.append(gen_loss)
        gen_optim.step()

        num_iters += 1
        if num_iters % NUM_STEPS_WRITE_REPORT:
            curr_time = time.time()
            elapsed_time_hrs, elapsed_time_mins, elapsed_time_secs = get_elapsed_time(start_time, curr_time)
            gen_losses_mean = np.mean(gen_losses)
            dis_losses_mean = np.mean(dis_losses)
            print("Trained the GAN for Atari games image synthesis for %d iterations in %d hours, %d mins, and %d secs: Generator_Loss=%.6f  Discriminator_Loss=%.6f" \
                   % (num_iters, elapsed_time_hrs, elapsed_time_mins, elapsed_time_secs, gen_losses_mean, dis_losses_mean))

            writer.add_scalar("Generator Loss", gen_losses_mean, num_iters)
            writer.add_scalar("Discriminator Loss", dis_losses_mean, num_iters)
            gen_losses = []
            dis_losses = []

        if num_iters % NUM_STEPS_SAVE_IMAGES:
            writer.add_image()
            writer.add_image()

        if num_iters % MAX_NUM_ITERS == 0:
            break

    curr_time = time.time()
    elapsed_time_hrs, elapsed_time_mins, elapsed_time_secs = get_elapsed_time(start_time, curr_time)
    print("Trained the Generative Adversarial Network (GAN) for Atari games image synthesis for %d iterations in %d hours, %d mins, and %d secs" \
          % (num_iters, elapsed_time_hrs, elapsed_time_mins, elapsed_time_secs))


if __name__ == "__main__":
    atari_games_env_list = ["ALE/Breakout-v5", "ALE/Alien-v5", "ALE/Atlantis-v5", "ALE/Robotank-v5", "ALE/Pitfall-v5", \
                            "ALE/VideoCube-v5", "ALE/VideoCheckers-v5", "ALE/BattleZone-v5", "ALE/Qbert-v5", "ALE/KungFuMaster-v5"]

    args = ArgumentParser()
    args.add_argument("--num_envs", default=10, description="Subset of number of games used to train the model.")
    num_envs = max(1, min(args.num_envs, len(atari_games_env_list)))
    envs = [InputWrapper(gym.make(env)) for env in atari_games_env_list]

    if torch.cuda.is_available():
        # If you have multiple GPUs, you can assign specific GPU via indices, for example `cuda:0`, `cuda:1`.
        device = torch.device("cuda")
    elif torch.mps.is_available():
        # For Apple's Silicon chip beginning from M2 with support for Metal Performance Shaders (mps).
        device = torch.device("mps")
    else:
        # Just use CPU if you don't have either GPU support or an Apple Computer with M2+ series chips that come with MPS support already enabled.
        device = torch.device("cpu")

    writer = SummaryWriter()
    train(envs, writer, device)