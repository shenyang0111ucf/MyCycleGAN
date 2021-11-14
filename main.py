import torch
import torch.nn as nn
import torch.optim as optim
import discriminator
import generator
import dataset
import config
import util
from tqdm import tqdm
from torchvision.utils import save_image

from torch.utils.data import DataLoader

train_dataset = dataset.GANDataset('D:\\CycleGAN_Data\\horse2zebra\\train\\A',
                                   'D:\\CycleGAN_Data\\horse2zebra\\train\\B',
                                   transform=config.transforms)
test_dataset = dataset.GANDataset('D:\\CycleGAN_Data\\horse2zebra\\test\\A',
                                  'D:\\CycleGAN_Data\\horse2zebra\\test\\B',
                                  transform=config.transforms)

epoch = 200
lr = 0.0002
batch_size = 4
num_workers = 8
device = config.DEVICE

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)


def lambda_rule(x):
    if x < 100:
        return lr
    else:
        return lr - (x-99)*lr/100


img_channels = 3
img_size = 256
lossfunc_mse = nn.MSELoss()
lossfunc_L1 = nn.L1Loss()

generator1 = generator.Generator(img_channels, num_residuals=9).to(device=device)
discriminator1 = discriminator.Discriminator(input_channels=img_channels).to(device=device)

generator2 = generator.Generator(img_channels, num_residuals=9).to(device=device)
discriminator2 = discriminator.Discriminator(input_channels=img_channels).to(device=device)
print('Finish loading data!')

optimizer_gen = optim.Adam(list(generator1.parameters()) + list(generator2.parameters()),
                           lr=lr, betas=(0.5, 0.999))
scheduler_gen = optim.lr_scheduler.LambdaLR(optimizer_gen, lr_lambda=lambda_rule)

optimizer_dis = optim.Adam(list(discriminator1.parameters()) + list(discriminator2.parameters()),
                           lr=lr, betas=(0.5, 0.999))
scheduler_dis = optim.lr_scheduler.LambdaLR(optimizer_dis, lr_lambda=lambda_rule)

scaler_gen = torch.cuda.amp.GradScaler()
scaler_dis = torch.cuda.amp.GradScaler()

print('finish creating models')


def test(generator1, generator2, device, test_loader, type1_name, type2_name):
    generator1.eval()
    generator2.eval()
    num = 1
    for batch_idx, (type1_img, type2_img) in enumerate(test_loader):
        type1_img = type1_img.to(device)
        type2_img = type2_img.to(device)
        type1_generated = generator2(type2_img)
        type2_generated = generator1(type1_img)

        save_image(type1_generated * 0.5 + 0.5, f"saved_images\\{type1_name}_{num}.png")
        save_image(type2_generated * 0.5 + 0.5, f"saved_images\\{type2_name}_{num+1}.png")
        num = num + 2


def train(generator1, discriminator1, generator2, discriminator2
        , device, train_loader, scheduler_gen, scheduler_dis
        ,optimizer_gen, optimizer_dis
        ,scaler_gen, scaler_dis
        ,criterion_mse, criterison_L1, batch_size, epoch_idx):
    generator1.train()
    discriminator1.train()
    generator2.train()
    discriminator2.train()
    for batch_idx, (type1_img, type2_img) in enumerate(train_loader):
        print(batch_idx)
        type1_img, type2_img = type1_img.to(device), type2_img.to(device)
        #original_sample_labels = torch.ones(batch_size, 1).to(device)
        #generated_samples_labels = torch.zeros(batch_size, 1).to(device)

        with torch.cuda.amp.autocast():
            generator1_output = generator1(type1_img)
            #samples = torch.cat((type2_img, generator1_output.detach()), 0)
            #samples_labels = torch.cat((original_sample_labels, generated_samples_labels), 0)

            discriminator2_output1 = discriminator2(type2_img)
            discriminator2_output2 = discriminator2(generator1_output.detach())
            discriminator2_output = torch.cat((discriminator2_output1, discriminator2_output2), 0)
            samples_labels = torch.cat((torch.ones_like(discriminator2_output1), torch.zeros_like(discriminator2_output2)), 0)

            loss_discriminator2 = criterion_mse(discriminator2_output, samples_labels)

            discriminator1.zero_grad()
            generator2_output = generator2(type2_img)
            #samples = torch.cat((type1_img, generator2_output.detach()), 0)

            discriminator1_output1 = discriminator1(type1_img)
            discriminator1_output2 = discriminator1(generator2_output.detach())
            discriminator1_output = torch.cat((discriminator1_output1, discriminator1_output2), 0)
            samples_labels = torch.cat((torch.ones_like(discriminator1_output1), torch.zeros_like(discriminator1_output2)), 0)

            loss_discriminator1 = criterion_mse(discriminator1_output, samples_labels)

            loss_discriminator = (loss_discriminator1 + loss_discriminator2) / 2
        optimizer_dis.zero_grad()
        scaler_dis.scale(loss_discriminator).backward()
        scaler_dis.step(optimizer_dis)
        scale_dis = scaler_dis.get_scale()
        scaler_dis.update()
        skip_scheduler_dis = (scale_dis != scaler_dis.get_scale())

#######################################################################################################
        with torch.cuda.amp.autocast():
            discriminator2_output = discriminator2(generator1_output)
            loss_generator1 = criterion_mse(discriminator2_output, torch.ones_like(discriminator2_output))

            generator1_compare = generator2(generator1_output)
            loss_generator1_compare = criterison_L1(type1_img, generator1_compare)

            discriminator1_output = discriminator1(generator2_output)
            loss_generator2 = criterion_mse(discriminator1_output, torch.ones_like(discriminator1_output))

            generator2_compare = generator1(generator2_output)
            loss_generator2_compare = criterison_L1(type2_img, generator2_compare)

            loss_generator = loss_generator1 + loss_generator2 + loss_generator1_compare * config.LAMBDA_CYCLE + loss_generator2_compare * config.LAMBDA_CYCLE
        optimizer_gen.zero_grad()
        scaler_gen.scale(loss_generator).backward()
        scaler_gen.step(optimizer_gen)
        scale_gen = scaler_gen.get_scale()
        scaler_gen.update()
        skip_scheduler_gen = (scale_gen != scaler_gen.get_scale())

    print(f"Epoch: {epoch_idx} Loss D.: {loss_discriminator}")
    print(f"Epoch: {epoch_idx} Loss G.: {loss_generator}")

    if not skip_scheduler_dis:
        scheduler_dis.step()
    if not skip_scheduler_gen:
        scheduler_gen.step()


if config.LOAD_MODEL:
    util.load_checkpoint(
        config.SAVE_DIR, config.CHECKPOINT_GEN_1, generator1, optimizer_gen, config.LEARNING_RATE,
    )
    util.load_checkpoint(
        config.SAVE_DIR, config.CHECKPOINT_GEN_2, generator2, optimizer_gen, config.LEARNING_RATE,
    )
    util.load_checkpoint(
        config.SAVE_DIR, config.CHECKPOINT_DIS_1, discriminator1, optimizer_dis, config.LEARNING_RATE,
    )
    util.load_checkpoint(
        config.SAVE_DIR, config.CHECKPOINT_DIS_2, discriminator2, optimizer_dis, config.LEARNING_RATE,
    )


if __name__ == '__main__':
    for i in range(epoch):
        print('epoch: ', i+1)

        train(generator1, discriminator1, generator2, discriminator2
              , device, train_loader, scheduler_gen, scheduler_dis
              , optimizer_gen, optimizer_dis
              , scaler_gen, scaler_dis
              , lossfunc_mse, lossfunc_L1, batch_size, i+1)
        if config.SAVE_MODEL:
            util.save_checkpoint(generator1, optimizer_gen, config.SAVE_DIR, filename=config.CHECKPOINT_GEN_1)
            util.save_checkpoint(generator2, optimizer_gen, config.SAVE_DIR, filename=config.CHECKPOINT_GEN_2)
            util.save_checkpoint(discriminator1, optimizer_dis, config.SAVE_DIR, filename=config.CHECKPOINT_DIS_1)
            util.save_checkpoint(discriminator2, optimizer_dis, config.SAVE_DIR,  filename=config.CHECKPOINT_DIS_2)

    type1_name = 'horse'
    type2_name = 'zebra'
    test(generator1, generator2, device, test_loader, type1_name, type2_name)


