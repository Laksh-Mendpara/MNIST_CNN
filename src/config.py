import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 9e-3
BATCH_SIZE = 128
NUM_EPOCHS = 15
NUM_WORKERS = 2
SCHEDULER_PATIENCE = 0
SCHEDULER_FACTOR = 0.4
MAX_NORM = 0.8
LOAD_MODEL = False
SAVE_MODEL = False
MODE = "min"
SCHEDULER = "RON"
CHECKPOINT_DIR = "my_checkpoint.pth.tar"
WANDB_PROJECT_NAME = "MNIST cnn fin"
# WANDB_ENTITY_NAME = "try06 param - 10700"
WANDB_CONFIG = {
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "scheduler_patience": SCHEDULER_PATIENCE,
    "scheduler": SCHEDULER_FACTOR
}
WANDB_LOG_FREQ = 10

train_transform = A.Compose(
    [
        A.Resize(height=32, width=32),
        A.RandomCrop(height=28, width=28),
        A.Normalize(mean=[0.5,], std=[0.5,], max_pixel_value=255.0),
        ToTensorV2()
    ]
)

test_transform = A.Compose(
    [
        A.Resize(height=32, width=32),
        A.CenterCrop(height=28, width=28),
        A.Normalize(mean=[0.5,], std=[0.5,], max_pixel_value=255.0),
        ToTensorV2()
    ]
)
