import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from tqdm import tqdm
import wandb
from model import CNN
import config
import utils
from dataset import MNISTDataset
import math

# Initializing wandb
# wandb.init(project=config.WANDB_PROJECT_NAME,
#            name=utils.create_wandb_name(config.WANDB_CONFIG, config.MODE, config.SCHEDULER, "8092"),
#            config=config.WANDB_CONFIG)

# Preparing Dataset
train_dataset = datasets.MNIST(root='dataset/', train=True, download=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, download=True)

train_dataset = MNISTDataset(data=train_dataset, transform=config.train_transform)
test_dataset = MNISTDataset(data=test_dataset, transform=config.test_transform)

train_loader = DataLoader(dataset=train_dataset, shuffle=True,
                          batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, drop_last=True)
test_loader = DataLoader(dataset=test_dataset, shuffle=False,
                         batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)

for data, targets in train_loader:
    print("input image shape :", data.shape)
    print("target shape :", targets.shape)
    break

# For storing logs
record = {
    "Train Loss": [],
    "Test Loss": [],
    "Train Accuracy": [],
    "Test Accuracy": []
}


# Train_function for a single epoch
def train_function(model, optimizer, scheduler, criterion, epoch):
    model.train()
    loop = tqdm(train_loader)

    for idx, (img, target) in enumerate(loop):
        img = img.to(config.DEVICE)
        target = target.to(config.DEVICE)

        outputs = model(img)
        loss = criterion(outputs, target)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_NORM)

        optimizer.step()

        if idx % 10 == 0:
            loop.set_description(f"Epoch: [{epoch+1}/{config.NUM_EPOCHS}]")
            loop.set_postfix(train_loss=loss.item())

    with torch.no_grad():
        train_loss = utils.compute_loss(model, criterion, train_loader)
        test_loss = utils.compute_loss(model, criterion, test_loader)

        train_accuracy = utils.compute_accuracy(model, train_loader)
        test_accuracy = utils.compute_accuracy(model, test_loader)

        record["Train Loss"].append(train_loss)
        record["Test Loss"].append(test_loss)
        record["Train Accuracy"].append(train_accuracy)
        record["Test Accuracy"].append(test_accuracy)

        print(f"Epoch: [{epoch+1}/{config.NUM_EPOCHS}] || train_loss: {train_loss:.4f} || test_loss: {test_loss:.4f}",
              end="")
        print(f" || train_acc: {train_accuracy:.2f} || test_acc: {test_accuracy:.2f}")

        metrics = {
            "Train Loss": train_loss,
            "Test Loss": test_loss,
            "Train Accuracy": train_accuracy,
            "Test Accuracy": test_accuracy
        }

        # wandb.log(metrics, step=epoch+1)

        if config.SCHEDULER == "RON" and config.MODE == "max":
            scheduler.step(test_accuracy)
        elif config.SCHEDULER == "RON" and config.MODE == "min":
            scheduler.step(test_loss)
        elif config.SCHEDULER == "cosine":
            scheduler.step()

    if config.SAVE_MODEL:
        utils.save_checkpoint(model, optimizer, config.CHECKPOINT_DIR)


# Main function
def main():
    print("Running on Device -", config.DEVICE)
    model = CNN().to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, betas=(0.86, 0.999))
    criterion = nn.CrossEntropyLoss()
    if config.LOAD_MODEL:
        print('loading pre-trained model')
        utils.load_checkpoint(config.CHECKPOINT_DIR, model, optimizer, config.LEARNING_RATE)
        train_accuracy = utils.compute_accuracy(model, train_loader)
        test_accuracy = utils.compute_accuracy(model, test_loader)
        print(f'loaded model: train_acc: {train_accuracy:.2f} || test_acc: {test_accuracy:.2f}"')

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode=config.MODE,
                                               patience=config.SCHEDULER_PATIENCE, factor=config.SCHEDULER_FACTOR)

    if scheduler == "cosine":
        q = math.floor(len(train_dataset)/config.BATCH_SIZE)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=q)

    # wandb.watch(model, criterion, log="all", log_freq=config.WANDB_LOG_FREQ)

    for epoch in range(config.NUM_EPOCHS):
        train_function(model, optimizer, scheduler, criterion, epoch)


if __name__ == "__main__":
    main()
    utils.plot_graph(record["Train Loss"], record["Test Loss"], label="Loss")
    utils.plot_graph(record["Train Accuracy"], record["Test Accuracy"], label="Accuracy")
