import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


from glob import glob
import numpy as np
import copy
import random
import time

import matplotlib.pyplot as plt
from PIL import Image
import cv2
# pip install pytorch_lightning/albumentations
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
# from torchmetrics.functional import (psnr, ssim, accuracy)
from sklearn.model_selection import train_test_split
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from os import listdir
import pandas as pd

from pip._internal import main
#main(['install','pytorch_lightning'])
#main(['install','albumentations'])




torch.manual_seed(0)
np.random.seed(0)

ORI_PATH = os.getcwd()
os.chdir("..")
PATH = os.getcwd() + os.path.sep + 'data'
folder = os.listdir(PATH)
base_path=PATH
PATH_AUG = os.getcwd() + os.path.sep + 'data_aug'
if not os.path.isdir(PATH_AUG):
    print('The directory is not present. Creating a new one..')
    os.mkdir(PATH_AUG)
else:
    print('The directory is present.')


PATH_working = os.getcwd() + os.path.sep + 'working'
if not os.path.isdir(PATH_working):
    print('The directory is not present. Creating a new one..')
    os.mkdir(PATH_working)
else:
    print('The directory is present.')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def extract_coords(df):
    coord = df.path.str.rsplit("_", n=4, expand=True)
    coord = coord.drop([0, 1, 4], axis=1)
    coord = coord.rename({2: "x", 3: "y"}, axis=1)
    coord.loc[:, "x"] = coord.loc[:,"x"].str.replace("x", "", case=False).astype(np.int)
    coord.loc[:, "y"] = coord.loc[:,"y"].str.replace("y", "", case=False).astype(np.int)
    df.loc[:, "x"] = coord.x.values
    df.loc[:, "y"] = coord.y.values
    return df

def get_cancer_dataframe(patient_id, cancer_id):
    path = base_path +"/" + patient_id + "/" + cancer_id
    files = listdir(path)
    dataframe = pd.DataFrame(files, columns=["filename"])
    path_names = path + "/" + dataframe.filename.values
    dataframe = dataframe.filename.str.rsplit("_", n=4, expand=True)
    dataframe.loc[:, "target"] = np.int(cancer_id)
    dataframe.loc[:, "path"] = path_names
    dataframe = dataframe.drop([0, 1, 4], axis=1)
    dataframe = dataframe.rename({2: "x", 3: "y"}, axis=1)
    dataframe.loc[:, "x"] = dataframe.loc[:,"x"].str.replace("x", "", case=False).astype(np.int)
    dataframe.loc[:, "y"] = dataframe.loc[:,"y"].str.replace("y", "", case=False).astype(np.int)
    return dataframe

def get_patient_dataframe(patient_id):
    df_0 = get_cancer_dataframe(patient_id, "0")
    df_1 = get_cancer_dataframe(patient_id, "1")
    patient_df = df_0.append(df_1)
    return patient_df


#example = get_patient_dataframe(data.patient_id.values[0])

#### augmentation

cfg = dict(
    seed=2022,
    train_dir="../input/histopathologic-cancer-detection/train/",
    # val_dir = "../input/histopathologic-cancer-detection/test/",
    num_images=10000,
    image_size=256,

    num_epochs=16,
    batch_size=4,  # the original paper used instance normalization i.e. bs = 1
    lr=2e-4,
    display_step=5,
    adversarial_criterion=nn.BCEWithLogitsLoss(),
    recon_criterion=nn.L1Loss(),
    lambda_recon=100,  # the original paper lambda = 100
)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(cfg['seed'])


both_transform = A.Compose(
    [
        A.Resize(width=cfg['image_size'], height=cfg['image_size']),   # default INTER_LINEAR interpolation
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ], additional_targets={"image0": "image"},
)



class CancerDataset(Dataset):
    #def __init__(self, df, base_dir, transform=None):
    def __init__(self, df, transform=None):
        #self.base_dir = base_dir
        # self.list_files = os.listdir(self.base_dir)
        #self.list_files = glob(os.path.join(self.base_dir, "*.tif"))[:cfg['num_images']]
        self.states = df[:cfg['num_images']]
        self.transform = transform

    def __len__(self):
        return len(self.states)
        #return len(self.list_files)

    def __getitem__(self, index):
        #img_path = self.list_files[index]
        img_path = self.states.path.values[index]
        img = cv2.imread(img_path)
        color_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # convert to 3-channel grayscale
        gray_image = cv2.cvtColor(cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY),
                                  cv2.COLOR_GRAY2RGB)

        if self.transform is not None:
            transformed = self.transform(image=color_image, image0=gray_image)
            color_image = transformed["image"]  # real image/target
            gray_image = transformed["image0"]  # conditioned image/input

        return color_image, gray_image

def visualize_original_images(df, sample=10):
    # display 10 images
    fig = plt.figure(figsize=(15, 5))
    # train_imgs = os.listdir(dir)[:cfg['num_images']]
    # train_imgs = glob(os.path.join(dir, "*.tif"))
    img_path = df.path.values
    for idx, img in enumerate(np.random.choice(img_path, sample)):
        ax = fig.add_subplot(2, sample//2, idx+1, xticks=[], yticks=[])
        im = Image.open(img)
        plt.imshow(im)
    plt.show()



def visualize_augmentations(dataset,samples=5):
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([t for t in dataset.transform
                                   if not isinstance(t, (A.Normalize, ToTensorV2))])
    figure, ax = plt.subplots(nrows=samples, ncols=2, figsize=(10, 20))

    for idx, i in enumerate(np.random.choice(cfg['num_images'], samples)):
        color_image, gray_image = dataset[i]
        ax[idx, 0].imshow(color_image)
        ax[idx, 1].imshow(gray_image)
        ax[idx, 0].set_title("color")
        ax[idx, 1].set_title("gray")
        ax[idx, 0].set_axis_off()
        ax[idx, 1].set_axis_off()
    plt.suptitle('Augmented images', y=1.0, fontsize=18)
    plt.tight_layout()
    plt.show()



class UpSampleConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel=4,
        strides=2,
        padding=1,
        activation=True,
        batchnorm=True,
        dropout=False
    ):
        super().__init__()
        self.activation = activation
        self.batchnorm = batchnorm
        self.dropout = dropout

        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel, strides, padding)

        if batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)

        if activation:
            self.act = nn.ReLU(True)

        if dropout:
            self.drop = nn.Dropout2d(0.5)

    def forward(self, x):
        x = self.deconv(x)
        if self.batchnorm:
            x = self.bn(x)

        if self.dropout:
            x = self.drop(x)
        return x

class DownSampleConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel=4, strides=2, padding=1, activation=True, batchnorm=True):
        """
        Paper details:
        - C64-C128-C256-C512-C512-C512-C512-C512
        - All convolutions are 4×4 spatial filters applied with stride 2
        - Convolutions in the encoder downsample by a factor of 2
        """
        super().__init__()
        self.activation = activation
        self.batchnorm = batchnorm

        self.conv = nn.Conv2d(in_channels, out_channels, kernel, strides, padding)

        if batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)

        if activation:
            self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        if self.batchnorm:
            x = self.bn(x)
        if self.activation:
            x = self.act(x)
        return x

class UNetGenerator(nn.Module):

    def __init__(self, in_channels, out_channels):
        """
        Paper details:
        - Encoder: C64-C128-C256-C512-C512-C512-C512-C512
        - All convolutions are 4×4 spatial filters applied with stride 2
        - Convolutions in the encoder downsample by a factor of 2
        - Decoder: CD512-CD1024-CD1024-C1024-C1024-C512 -C256-C128
        """
        super().__init__()

        # encoder/donwsample convs
        self.encoders = [
            DownSampleConv(in_channels, 64, batchnorm=False),  # bs x 64 x 128 x 128
            DownSampleConv(64, 128),  # bs x 128 x 64 x 64
            DownSampleConv(128, 256),  # bs x 256 x 32 x 32
            DownSampleConv(256, 512),  # bs x 512 x 16 x 16
            DownSampleConv(512, 512),  # bs x 512 x 8 x 8
            DownSampleConv(512, 512),  # bs x 512 x 4 x 4
            DownSampleConv(512, 512),  # bs x 512 x 2 x 2
            DownSampleConv(512, 512, batchnorm=False),  # bs x 512 x 1 x 1
        ]

        # decoder/upsample convs
        self.decoders = [
            UpSampleConv(512, 512, dropout=True),  # bs x 512 x 2 x 2
            UpSampleConv(1024, 512, dropout=True),  # bs x 512 x 4 x 4
            UpSampleConv(1024, 512, dropout=True),  # bs x 512 x 8 x 8
            UpSampleConv(1024, 512),  # bs x 512 x 16 x 16
            UpSampleConv(1024, 256),  # bs x 256 x 32 x 32
            UpSampleConv(512, 128),  # bs x 128 x 64 x 64
            UpSampleConv(256, 64),  # bs x 64 x 128 x 128
        ]
        self.decoder_channels = [512, 512, 512, 512, 256, 128, 64]
        self.final_conv = nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

    def forward(self, x):
        skips_cons = []
        for encoder in self.encoders:
            x = encoder(x)

            skips_cons.append(x)

        skips_cons = list(reversed(skips_cons[:-1]))
        decoders = self.decoders[:-1]

        for decoder, skip in zip(decoders, skips_cons):
            x = decoder(x)
            # print(x.shape, skip.shape)
            x = torch.cat((x, skip), axis=1)

        x = self.decoders[-1](x)
        # print(x.shape)
        x = self.final_conv(x)
        return self.tanh(x)

class PatchGANDiscriminator(nn.Module):

    def __init__(self, input_channels):
        super().__init__()
        self.d1 = DownSampleConv(input_channels, 64, batchnorm=False)
        self.d2 = DownSampleConv(64, 128)
        self.d3 = DownSampleConv(128, 256)
        self.d4 = DownSampleConv(256, 512)
        self.final = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, x, y):
        x = torch.cat([x, y], axis=1)
        x0 = self.d1(x)
        x1 = self.d2(x0)
        x2 = self.d3(x1)
        x3 = self.d4(x2)
        xn = self.final(x3)
        return xn


def _weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


def display_progress(cond, real, fake, current_epoch, path_process,figsize=(10, 5)):
    """
    Save cond, real (original) and generated (fake)
    images in one panel
    """
    cond = cond.detach().cpu().permute(1, 2, 0)
    real = real.detach().cpu().permute(1, 2, 0)
    fake = fake.detach().cpu().permute(1, 2, 0)

    images = [cond, real, fake]
    titles = ['input', 'real', 'generated']
    print(f'Epoch: {current_epoch}')
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    for idx, img in enumerate(images):
        ax[idx].imshow(img)
        ax[idx].axis("off")
    for idx, title in enumerate(titles):
        ax[idx].set_title('{}'.format(title))
    fig.suptitle('Epcoh {}'.format(current_epoch))
    plt.savefig(path_process)
    plt.show()




class Pix2Pix(pl.LightningModule):

    def __init__(self, in_channels, out_channels, learning_rate=0.0002, lambda_recon=100, display_step=5):

        super().__init__()
        self.save_hyperparameters()

        self.display_step = display_step
        self.gen = UNetGenerator(in_channels, out_channels)
        self.disc = PatchGANDiscriminator(in_channels + out_channels)

        # intializing weights
        self.gen = self.gen.apply(_weights_init)
        self.disc = self.disc.apply(_weights_init)

        self.adversarial_criterion = cfg['adversarial_criterion']  # nn.BCEWithLogitsLoss()
        self.recon_criterion = cfg['recon_criterion']  # nn.L1Loss()

    def _gen_step(self, real_images, conditioned_images):
        # Pix2Pix has adversarial and a reconstruction loss
        # adversarial loss
        fake_images = self.gen(conditioned_images)
        disc_logits = self.disc(fake_images, conditioned_images)
        adversarial_loss = self.adversarial_criterion(disc_logits, torch.ones_like(disc_logits))

        # reconstruction loss
        recon_loss = self.recon_criterion(fake_images, real_images)
        lambda_recon = self.hparams.lambda_recon

        return adversarial_loss + lambda_recon * recon_loss

    def _disc_step(self, real_images, conditioned_images):
        fake_images = self.gen(conditioned_images).detach()
        fake_logits = self.disc(fake_images, conditioned_images)

        real_logits = self.disc(real_images, conditioned_images)

        fake_loss = self.adversarial_criterion(fake_logits, torch.zeros_like(fake_logits))
        real_loss = self.adversarial_criterion(real_logits, torch.ones_like(real_logits))
        return (real_loss + fake_loss) / 2

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr)
        disc_opt = torch.optim.Adam(self.disc.parameters(), lr=lr)
        return disc_opt, gen_opt

    # the optimizer index is used to index multiple (here two) optimizers
    def training_step(self, batch, batch_idx, optimizer_idx):
        real, condition = batch

        loss = None
        if optimizer_idx == 0:
            loss = self._disc_step(real, condition)
            self.log('PatchGAN Loss', loss)
        elif optimizer_idx == 1:
            loss = self._gen_step(real, condition)
            self.log('Generator Loss', loss)

        if self.current_epoch >=0  and batch_idx == 0 and optimizer_idx == 1:
            fake = self.gen(condition).detach()
            display_progress(condition[0], real[0], fake[0], self.current_epoch,
                             path_process=PATH_working+"/process_img_{}".format(self.current_epoch))


        if self.current_epoch % self.display_step >= 0 :

            torch.save(self.gen.state_dict(), "generator_model_{}.pt".format(self.current_epoch))
            torch.save(self.disc.state_dict(), "discriminator_model_{}.pt".format(self.current_epoch))


        return loss


if __name__ == '__main__':

    total_images = 0
    for n in range(len(folder)):
        patient_id = folder[n]
        for c in [0, 1]:
            patient_path = base_path + "/" + patient_id
            class_path = patient_path + "/" + str(c) + "/"
            subfiles = listdir(class_path)
            total_images += len(subfiles)

    data = pd.DataFrame(index=np.arange(0, total_images), columns=["patient_id", "path", "target"])

    k = 0
    for n in range(len(folder)):
        patient_id = folder[n]
        patient_path = base_path + "/" + patient_id
        for c in [0, 1]:
            class_path = patient_path + "/" + str(c) + "/"
            subfiles = listdir(class_path)
            for m in range(len(subfiles)):
                image_path = subfiles[m]
                data.iloc[k]["path"] = class_path + image_path
                data.iloc[k]["target"] = c
                data.iloc[k]["patient_id"] = patient_id
                k += 1

    data.loc[:, "target"] = data.target.astype(np.str)

    # split the train and test file (train 70%, test 15%, val 15%)
    patients = data.patient_id.unique()

    train_ids, sub_test_ids = train_test_split(patients,
                                               test_size=0.3,
                                               random_state=0)
    test_ids, dev_ids = train_test_split(sub_test_ids, test_size=0.5, random_state=0)

    train_df = data.loc[data.patient_id.isin(train_ids), :].copy()
    test_df = data.loc[data.patient_id.isin(test_ids), :].copy()
    dev_df = data.loc[data.patient_id.isin(dev_ids), :].copy()

    train_df = extract_coords(train_df)
    test_df = extract_coords(test_df)
    dev_df = extract_coords(dev_df)



    train_dataset = CancerDataset(train_df, transform=both_transform)
    dataloader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
    #visualize_original_images(train_df, 10)
    #visualize_augmentations(train_dataset)

    pix2pix = Pix2Pix(3, 3, learning_rate=cfg['lr'], lambda_recon=cfg['lambda_recon'],
                      display_step=cfg['display_step'])
    trainer = pl.Trainer(max_epochs=cfg['num_epochs'], gpus=-1)
    trainer.fit(pix2pix, dataloader)

    print('generate success!')






















