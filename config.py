import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "horse2zebra/train"
VAL_DIR = "horse2zebra/test"
BATCH_SIZE = 8
LEARNING_RATE = 1e-5
LAMBDA_CYCLE = 10
LAMBDA_IDENTITY = 0.5
LAMBDA_FEATURE = 5
NUM_WORKERS = 16
NUM_EPOCHS = 200
SAVE_DIR = "pretrained_model/"
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_1 = "gen1.pth.tar"
CHECKPOINT_GEN_2 = "gen2.pth.tar"
CHECKPOINT_DIS_1 = "dis1.pth.tar"
CHECKPOINT_DIS_2 = "dis2.pth.tar"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)