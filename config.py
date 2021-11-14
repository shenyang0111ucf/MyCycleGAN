import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
BATCH_SIZE = 1
LEARNING_RATE = 0.0002
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 10
SAVE_DIR = "pretrained_model\\"
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