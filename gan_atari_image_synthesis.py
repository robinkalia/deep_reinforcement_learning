import argparse
import cv2
import numpy as np

import typing as tt
from typing import List, Tuple

import torch
from torch import nn, optim
from torch.utils.tensorboard.writer import SummaryWriter
import torch.utils as vutils

import gymnasium as gym
from gymnasium import spaces

import random
import time

import torchvision.utils as visionutils

LATENT_VEC_SIZE = 100

PROCESSING_WIDTH = 64
PROCESSING_HEIGHT = 64

NUM_DISCR_CHANNELS = 128
NUM_GENER_CHANNELS = 128

BATCH_SIZE = 16

IMAGE_MEAN_THRESH = 0.01

NUM_STEPS_WRITE_REPORT = 100
NUM_STEPS_SAVE_IMAGES = 1000

NUM_IMAGES_GRID_ROW = 16

MAX_NUM_ITERATIONS = 1000000

# Input wrapper for the environments whose purpose is to basically return images of original size: 210x160
# and resize them to a target size of 128x128 and return them.
class InputWrapper(gym.ObservationWrapper):
    def __init__(self, *args):
        super(InputWrapper, self).__init__(*args)
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
        # Converts the input image of size (3, 128, 128) into a final single floating point number at the end.
        self.discriminator_model = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=NUM_DISCR_CHANNELS, kernel_size=4, stride=2, padding=1),
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
            nn.Conv2d(in_channels=NUM_DISCR_CHANNELS*8, out_channels=1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        res = self.discriminator_model(x)
        print(f"Discriminator Model: x.shape={x.shape}\tres.shape={res.shape}")
        return res.view(-1, 1).squeeze(dim=1)


class Generator(nn.Module):
    def __init__(self, output_shape):
        super(Generator, self).__init__()
        # Converts a given vector to an image of size (3, 128, 128) at the end.
        self.generator_model = nn.Sequential(
            nn.ConvTranspose2d(in_channels=LATENT_VEC_SIZE, out_channels=NUM_GENER_CHANNELS*8, kernel_size=4, stride=1, padding=0),
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
        return res


def iterate_batches(envs: List[gym.Env], batch_size: int=BATCH_SIZE) -> tt.Generator[torch.Tensor, None, None]:
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


def train(envs: List[gym.Env], writer: SummaryWriter, device: torch.device):
    image_shape = envs[0].observation_space.shape
    print("Image Shape: ", image_shape)
    discriminator_network = Discriminator(input_shape=image_shape)
    discriminator_network = discriminator_network.to(device)

    generator_network = Generator(output_shape=image_shape)
    generator_network = generator_network.to(device)

    disc_optim = optim.Adam(discriminator_network.parameters(), lr=0.001, weight_decay=1e-4, betas=(0.9, 0.99))
    gen_optim = optim.Adam(generator_network.parameters(), lr=0.001, weight_decay=1e-4, betas=(0.9, 0.99))

    loss_func = nn.BCELoss()

    # There are 2 label values: 1 for real images that we grab from atari games' environment
    # and 0 for fake images that are generated by the generator network.
    generic_true_labels = torch.ones(BATCH_SIZE, device=device)
    generic_fake_labels = torch.zeros(BATCH_SIZE, device=device)

    generator_losses = []
    discriminator_losses = []

    start_time = time.time()

    num_iters = 0
    for batch_images in iterate_batches(envs):
        # The next 3 steps essentially define the training steps of a Generative Adversarial Network (GAN)
        # Step 1: Generate fake images from a random vector using the Generator network.
        rand_generator_input = torch.FloatTensor(BATCH_SIZE, LATENT_VEC_SIZE, 1, 1)
        rand_generator_input.normal_(0,1)
        rand_generator_input = rand_generator_input.to(device)
        fake_gen_images = generator_network(rand_generator_input)

        # Step 2: Teach the discriminator to identify real images from fake images.
        disc_optim.zero_grad()
        batch_images = batch_images.to(device)
        disc_pred_real_images = discriminator_network(batch_images)
        print(f"batch_images.shape={batch_images.shape}\nResult from Discriminator network-disc_pred_real_images.shape={disc_pred_real_images.shape}")
        # We create a copy of the generated images since we don't want the gradients to propagate as we use
        # the original tensor for generator images to the compute the generator loss in step 3 (next step).
        disc_pred_fake_images = discriminator_network(fake_gen_images.detach())
        # Calculate the combined loss on both real and fake images while giving them
        # the same weightage initially. However, we can tune this relative ratio as a hyperparameter.
        disc_loss = loss_func(generic_true_labels, disc_pred_real_images) +  \
                    loss_func(generic_fake_labels, disc_pred_fake_images)
        disc_loss.backward()
        discriminator_losses.append(disc_loss.item())
        disc_optim.step()

        # Step 3: Teach the generator to train the model to produce better fake images depending upon
        # how well the discriminator confused these fake images as real images.
        gen_optim.zero_grad()
        # Recompute the predictions from the updated discriminator network for the same fake images.
        updated_disc_pred_fake_images_recomputed = discriminator_network(fake_gen_images)
        # For the generator we have to check how further the discriminator is computing predictions
        # for these fake generated images from true labels. The generator wants the discriminator to
        # predict these fake images that it has genererated as true labels. That is why we compute
        # the generator loss as KL divergence between these discriminator predictions and true labels.
        gen_loss = loss_func(generic_true_labels, updated_disc_pred_fake_images_recomputed)
        gen_loss.backward()
        generator_losses.append(gen_loss.item())
        gen_optim.step()

        num_iters += 1
        if num_iters % NUM_STEPS_WRITE_REPORT:
            curr_time = time.time()
            elapsed_time_hrs, elapsed_time_mins, elapsed_time_secs = get_elapsed_time(start_time, curr_time)
            gen_losses_mean = np.mean(generator_losses)
            dis_losses_mean = np.mean(discriminator_losses)
            print("Trained the GAN for Atari games image synthesis for %d iterations in %d hours, %d mins, and %d secs: Generator_Loss=%.6f  Discriminator_Loss=%.6f" \
                   % (num_iters, elapsed_time_hrs, elapsed_time_mins, elapsed_time_secs, gen_losses_mean, dis_losses_mean))

            writer.add_scalar("Generator Loss", gen_losses_mean, num_iters)
            writer.add_scalar("Discriminator Loss", dis_losses_mean, num_iters)
            generator_losses = []
            discriminator_losses = []

        if num_iters % NUM_STEPS_SAVE_IMAGES:
            fake_images_grid = visionutils.make_grid(fake_gen_images.data[:64], nrow=NUM_IMAGES_GRID_ROW, padding=4, normalize=True)
            writer.add_image("Generated Images", fake_images_grid, num_iters)

            real_images_grid = visionutils.make_grid(batch_images.data[:64], nrow=NUM_IMAGES_GRID_ROW, padding=4, normalize=True)
            writer.add_image("Real Images", real_images_grid, num_iters)

        if num_iters % MAX_NUM_ITERATIONS == 0:
            break

    curr_time = time.time()
    elapsed_time_hrs, elapsed_time_mins, elapsed_time_secs = get_elapsed_time(start_time, curr_time)
    print("Trained the Generative Adversarial Network (GAN) for Atari games image synthesis for %d iterations in %d hours, %d mins, and %d secs" \
          % (num_iters, elapsed_time_hrs, elapsed_time_mins, elapsed_time_secs))


if __name__ == "__main__":
    atari_games_env_list = ["ALE/Breakout-v5", "ALE/Alien-v5", "ALE/Atlantis-v5", "ALE/Robotank-v5", "ALE/Pitfall-v5", \
                            "ALE/VideoCube-v5", "ALE/VideoCheckers-v5", "ALE/BattleZone-v5", "ALE/Qbert-v5", "ALE/KungFuMaster-v5"]
    num_atari_games_env = len(atari_games_env_list)

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", default=num_atari_games_env, help="Subset of number of games used to train the model.")
    args = parser.parse_args()

    num_envs = max(1, min(args.num_envs, len(atari_games_env_list)))
    envs = [InputWrapper(gym.make(env)) for env in atari_games_env_list[:num_envs]]

    if torch.cuda.is_available():
        # If you have multiple GPUs, you can assign specific GPU via indices, for example `cuda:0`, `cuda:1`.
        device = torch.device("cuda")
    elif torch.mps.is_available():
        # For Apple's Silicon chip beginning from M2 with support for Metal Performance Shaders (mps).
        device = torch.device("mps")
    else:
        # Just use CPU if you don't have either GPU support or an Apple Computer with M2+ series chips that come with MPS support already enabled.
        device = torch.device("cpu")

    print("Running the code on device:", device)

    writer = SummaryWriter()
    train(envs, writer, device)
    writer.close()
