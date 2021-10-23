import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from utils.config import *

# Creating logger and log file
from utils.dataset import ImageDataset
from utils.model import GeneratorResNet, Discriminator, FeatureExtractor

logging.basicConfig(filename=LOG_FILE_OUTPUT,
                    format='%(asctime)s %(message)s',
                    filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

try:
    os.mkdir(IMAGE_OUTPUT_PATH)
except FileExistsError as e:
    print(e)

try:
    os.mkdir(MODEL_SAVE_DIR)
except FileExistsError as e:
    print(e)

config = CONFIG

cuda = torch.cuda.is_available()
logger.info(f'Cuda is present - {cuda}')

hr_shape = (config['hr_height'], config['hr_width'])

generator = GeneratorResNet()
logger.info(generator)

discriminator = Discriminator(input_shape=(config['channels'], *hr_shape))
logger.info(discriminator)

feature_extractor = FeatureExtractor()
logger.info(feature_extractor.eval())

criterion_GAN = torch.nn.MSELoss()
criterion_content = torch.nn.L1Loss()

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    feature_extractor = feature_extractor.cuda()
    criterion_GAN = criterion_GAN.cuda()
    criterion_content = criterion_content.cuda()

if config['epoch'] != 0:
    generator.load_state_dict(torch.load(os.path.join(MODEL_SAVE_DIR, f'generator_{config["epoch"]}.pth')))
    discriminator.load_state_dict(torch.load(os.path.join(MODEL_SAVE_DIR, f'discriminator_{config["epoch"]}.pth')))

optimizer_G = torch.optim.Adam(generator.parameters(),
                               lr=config['lr'],
                               betas=(config['b1'],
                                      config['b2']))
optimizer_D = torch.optim.Adam(discriminator.parameters(),
                               lr=config['lr'],
                               betas=(config['b1'], config['b2']))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

dataloader = DataLoader(
    ImageDataset(f'{ROOT_DATA_DIR}/{config["dataset_name"]}', hr_shape=hr_shape),
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=config["n_cpu"],
)

for epoch in range(config["epoch"], config["n_epochs"]):
    for i, images in enumerate(dataloader):
        images_lr = Variable(images["lr"].type(Tensor))
        images_hr = Variable(images["hr"].type(Tensor))

        valid = Variable(Tensor(np.ones((images_lr.size(0),
                                         *discriminator.output_shape))),
                         requires_grad=False)
        fake = Variable(Tensor(np.zeros((images_lr.size(0),
                                         *discriminator.output_shape))),
                        requires_grad=False)

        # ---------------
        # Train Generator
        # ---------------

        optimizer_G.zero_grad()
        gen_hr = generator(images_lr)

        loss_GAN = criterion_GAN(discriminator(gen_hr), valid)

        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(images_hr)
        loss_content = criterion_content(gen_features, real_features.detach())

        loss_G = loss_content + 1e-3 * loss_GAN

        loss_G.backward()
        optimizer_G.step()

        # --------------------
        # Train Discriminator
        # --------------------

        optimizer_D.zero_grad()

        loss_real = criterion_GAN(discriminator(images_hr), valid)
        loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)

        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()

        # ---------------------
        # LOG PROGRESS
        # ---------------------

        logger.info(
            f'[EPOCH {epoch}/{config["n_epochs"]}] [BATCH {i}/{len(dataloader)}]'
        )
        logger.info(
            f"--------> [D LOSS : {loss_D.item()}] [G LOSS : {loss_G.item()}]"
        )

        batches_done = epoch * len(dataloader) + i

        if batches_done % config["sample_interval"] == 0:
            images_lr = nn.functional.interpolate(images_lr, scale_factor=4)
            gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
            images_lr = make_grid(images_lr, nrow=1, normalize=True)
            images_grid = torch.cat((images_lr, gen_hr), -1)
            save_image(images_grid, f"{IMAGE_OUTPUT_PATH}/{batches_done}.png", normalize=False)

        if config["checkpoint_interval"] != -1 and epoch % config["checkpoint_interval"] == 0:
            torch.save(generator.state_dict(),
                       f"{MODEL_SAVE_DIR}/generator_{epoch}.pth")
            torch.save(discriminator.state_dict(),
                       f"{MODEL_SAVE_DIR}/discriminator_{epoch}.pth")
