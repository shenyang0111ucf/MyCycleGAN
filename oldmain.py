import torch
from dataset import GANDataset
import sys
from util import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator import Discriminator
from generator import Generator


def train_fn(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, epoch_idx):
    H_reals = 0
    H_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (zebra, horse) in enumerate(loop):
        zebra = zebra.to(config.DEVICE)
        horse = horse.to(config.DEVICE)

        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_horse = gen_H(zebra)
            D_H_real = disc_H(horse)
            D_H_fake = disc_H(fake_horse.detach())
            H_reals += D_H_real.mean().item()
            H_fakes += D_H_fake.mean().item()
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss

            fake_zebra = gen_Z(horse)
            D_Z_real = disc_Z(zebra)
            D_Z_fake = disc_Z(fake_zebra.detach())
            D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
            D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            # put it togethor
            D_loss = (D_H_loss + D_Z_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_H_fake = disc_H(fake_horse)
            D_Z_fake = disc_Z(fake_zebra)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

            # cycle loss
            cycle_zebra = gen_Z(fake_horse)
            cycle_horse = gen_H(fake_zebra)
            cycle_zebra_loss = l1(zebra, cycle_zebra)
            cycle_horse_loss = l1(horse, cycle_horse)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_zebra = gen_Z(zebra)
            identity_horse = gen_H(horse)
            identity_zebra_loss = l1(zebra, identity_zebra)
            identity_horse_loss = l1(horse, identity_horse)

            # add all togethor
            G_loss = (
                    loss_G_Z
                    + loss_G_H
                    + cycle_zebra_loss * config.LAMBDA_CYCLE
                    + cycle_horse_loss * config.LAMBDA_CYCLE
                    + identity_horse_loss * config.LAMBDA_IDENTITY
                    + identity_zebra_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()
    print(f"Epoch: {epoch_idx} Loss D.: {D_loss}")
    print(f"Epoch: {epoch_idx} Loss G.: {G_loss}")
def test(generator1, generator2, device, test_loader, type1_name, type2_name):
    generator1.eval()
    generator2.eval()
    num = 1
    for batch_idx, (type1_img, type2_img) in enumerate(test_loader):
        type1_img = type1_img.to(device)
        type2_img = type2_img.to(device)
        type1_generated = generator2(type2_img)
        type2_generated = generator1(type1_img)

        save_image(type1_generated * 0.5 + 0.5, f"saved_images/{type1_name}_{num}.png")
        save_image(type2_generated * 0.5 + 0.5, f"saved_images/{type2_name}_{num+1}.png")
        num = num + 2

def main(testmode=False):
    disc_H = Discriminator(input_channels=3).to(config.DEVICE)
    disc_Z = Discriminator(input_channels=3).to(config.DEVICE)
    gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.SAVE_DIR, config.CHECKPOINT_GEN_1, gen_H, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.SAVE_DIR, config.CHECKPOINT_GEN_2, gen_Z, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.SAVE_DIR, config.CHECKPOINT_DIS_1, disc_H, opt_disc, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.SAVE_DIR, config.CHECKPOINT_DIS_2, disc_Z, opt_disc, config.LEARNING_RATE,
        )

    dataset = GANDataset(
        root_type1=config.TRAIN_DIR + "/A", root_type2=config.TRAIN_DIR + "/B", transform=config.transforms
    )
    val_dataset = GANDataset(
        root_type1=config.VAL_DIR + "/A", root_type2=config.VAL_DIR + "/B", transform=config.transforms
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    if not testmode:
        for epoch in range(config.NUM_EPOCHS):
            train_fn(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler, epoch)

            if config.SAVE_MODEL:
                save_checkpoint(gen_H, opt_gen, config.SAVE_DIR, filename=config.CHECKPOINT_GEN_1)
                save_checkpoint(gen_Z, opt_gen,config.SAVE_DIR,  filename=config.CHECKPOINT_GEN_2)
                save_checkpoint( disc_H, opt_disc, config.SAVE_DIR, filename=config.CHECKPOINT_DIS_1)
                save_checkpoint(disc_Z, opt_disc, config.SAVE_DIR, filename=config.CHECKPOINT_DIS_2)
    else:
        type1_name = 'horse'
        type2_name = 'zebra'
        test(gen_H, gen_Z, config.DEVICE, val_loader, type1_name, type2_name)
if __name__ == "__main__":
    main(False)