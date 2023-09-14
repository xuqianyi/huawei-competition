import argparse
import os
import numpy as np
import math

from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.nn.functional as F
import torch

from dataset_ptbxl import get_dataset
from cgan import Generator, Discriminator

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=12000, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=44, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=1000, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=12, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--data_path", type=str, default="/home/qianyi/huawei-competition/val.pt")
parser.add_argument("--model_path", type=str, default="/home/qianyi/huawei-competition/ckpt/model.pt")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size)

cuda = True if torch.cuda.is_available() else False

def save_checkpoints(self, checkpoint_path):
        save_dict = {}
        for model_name in self._model_names:
            save_dict.update({model_name: self.models[model_name].state_dict()})
        if isinstance(self.optim, dict):
            opt_dict = {optim_name: self.optim[optim_name].state_dict() for optim_name in self._opt_names}
            save_dict.update({"optim": opt_dict})
        else:
            save_dict.update({"optim": self.optim.state_dict()})
        torch.save(save_dict, checkpoint_path)

def load_checkpoints(self, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    for model_name in self._model_names:
        self.models[model_name].load_state_dict(checkpoint[model_name])
    if self.training:
        if isinstance(self.optim, dict):
            for opt_name in self._opt_names:
                self.optim[opt_name].load_state_dict(checkpoint["optim"][opt_name])
        else:
            self.optim.load_state_dict(checkpoint["optim"])

# Loss functions
adversarial_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = Generator(opt.img_size, opt.n_classes)
discriminator = Discriminator(opt.img_size, opt.n_classes)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

dataloader = get_dataset(opt.data_path)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):
        imgs = torch.unsqueeze(imgs[0], 0).cuda()
        labels = torch.unsqueeze(labels, 0).cuda()
        batch_size = imgs.shape[0]

        # Configure input
        real_imgs = imgs

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = torch.randn(batch_size, opt.img_size).cuda()
        gen_labels = torch.nn.functional.one_hot(torch.randint(0, opt.n_classes, (1,)), opt.n_classes).cuda()

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, torch.ones_like(validity))

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(real_imgs, labels)
        d_real_loss = adversarial_loss(validity_real, torch.ones_like(validity_real))

        # Loss for fake images
        validity_fake = discriminator(gen_imgs.detach(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake, torch.zeros_like(validity_fake))

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

    print(
        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
        % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
    )
    torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optG_state_dict': optimizer_G.state_dict(),
            'optD_state_dict': optimizer_D.state_dict(),
            }, opt.model_path)
