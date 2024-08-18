import torch
import matplotlib.pyplot as plt
import config


def compute_loss(model, criterion, loader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=config.DEVICE)
            y = y.to(device=config.DEVICE)

            output = model(x)
            loss = criterion(output, y)
            total_loss += loss

    model.train()
    return total_loss/float(len(loader))


def compute_accuracy(model, loader):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=config.DEVICE)
            y = y.to(device=config.DEVICE)

            output = model(x)
            _, predictions = output.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples * 100


def plot_graph(train_data, test_data, label):
    plt.plot(train_data, label=f"Train {str(label)}")
    plt.plot(test_data, label=f"Test {str(label)}")
    plt.xlabel("Epochs")
    plt.ylabel(label)
    plt.title(label + " vs Epochs")
    plt.legend()
    plt.savefig("./results/" + label + "_plot.png")
    plt.show()
    plt.clf()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    # print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def create_wandb_name(wandb_config, mode, scheduler, arch):
    wandb_name = "" if arch == "10700" else str(arch)
    for key, value in wandb_config.items():
        wandb_name += str(key) + ":" + str(value) + "__"
    if mode == "max":
        wandb_name += "mode:max__"
    if scheduler != "RON":
        wandb_name += "scheduler:" + str(scheduler)
    return wandb_name
