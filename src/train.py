import os
import argparse
import yaml
import multiprocessing as mp
import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from lib.mlp import MLP

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_data(**kwargs):
    # Load MNIST
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    valset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    trainloader = DataLoader(trainset, batch_size=kwargs["batch_size"], shuffle=True)
    testloader = DataLoader(valset, batch_size=kwargs["val_batch_size"], shuffle=False)
    return trainloader, testloader


def train(
    trainloader,
    testloader,
    valid=True,
    **kwargs,
):
    model = MLP(28 * 28, 10).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=kwargs["lr"])

    criterion = nn.CrossEntropyLoss()

    for _ in range(kwargs["epochs"]):
        # Train
        model.train()
        with tqdm(trainloader) as pbar:
            for i, (images, labels) in enumerate(pbar):
                optimizer.zero_grad()
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
        if valid:
            # Validation
            model.eval()
            val_loss, val_acc = 0, 0
            with torch.no_grad():
                for images, labels in testloader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    output = model(images)
                    val_loss += F.nll_loss(output, labels, reduction="sum").item()
                    val_acc += (output.argmax(dim=1) == labels).float().mean().item()
            val_acc /= len(testloader)
            mlflow.log_metric("val_loss", val_loss)
            mlflow.log_metric("val_acc", val_acc)

    return model


def worker_function(params, i, trainloader, testloader):
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("MNIST-test")
    with mlflow.start_run(nested=True):
        mlflow.set_tags(tags={"name": f"{params['name']}_{i}"})
        model = train(trainloader, testloader, **params)
        if params["save"]:
            mlflow.pytorch.log_model(model, "model")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_file", "-yml", type=str, default=None)
    parser.add_argument("--name", "-n", type=str, default="run")
    parser.add_argument("--batch_size", "-bs", type=int, default=64)
    parser.add_argument("--val_batch_size", "-vbs", type=int, default=1000)
    parser.add_argument("--epochs", "-e", type=int, default=1)
    parser.add_argument("--lr", "-lr", type=float, default=1e-3)
    parser.add_argument("--trials", "-t", type=int, default=1)
    parser.add_argument("--save", "-s", type=bool, default=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("MNIST-test")
    args = parse_arguments()
    params = vars(args)
    if params["yaml_file"]:
        with open(args.yaml_file, "r") as f:
            yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            for config in yaml_config["args"]:
                if config in params:
                    params[config] = yaml_config["args"][config]
    print(params)

    trainloader, testloader = load_data(**params)

    with mp.Pool(processes=1) as pool:
        pool.starmap(
            worker_function,
            [(params, i, trainloader, testloader) for i in range(params["trials"])],
        )
