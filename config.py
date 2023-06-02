import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.geometric.rotate import RandomRotate90

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/train/"
VAL_DIR = "data/val/"
PREPRO_DIR = "data/prepro/"
LEARNING_RATE = 2e-4
BATCH_SIZE = 4
VAL_BATCH_SIZE = 4
NUM_WORKERS = 4
IMAGE_SIZE = 128
CHANNELS_IMG = 1
L1_LAMBDA = 10
LAMBDA_GP = 10
NUM_EPOCHS = 251
VALIDATE = True
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_GEN = "model.pth.tar"

both_transform = A.Compose(
    [A.Resize(width=IMAGE_SIZE, height=IMAGE_SIZE),], additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [   RandomRotate90(),
        ToTensorV2()
    ]
)

transform_only_mask = A.Compose(
    [   RandomRotate90(),
        ToTensorV2()
    ]
)