import os
import argparse
import mlflow
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

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


def test(model, testloader):
    model.eval()
    val_loss, val_acc = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            output = model(images)
            val_loss += F.nll_loss(output, labels, reduction="sum").item()
            val_acc += (output.argmax(dim=1) == labels).float().mean().item()
    val_acc /= len(testloader)
    print(f"Val Loss: {val_loss}, Val Accuracy: {val_acc}")


def get_results(exp, metric, trials):
    client = mlflow.tracking.MlflowClient(os.environ["MLFLOW_TRACKING_URI"])
    res = []
    for i in range(trials):
        run = pd.DataFrame(
            mlflow.search_runs(
                search_all_experiments=True, filter_string=f"tags.name='{exp}_{i}'"
            )
        )
        id = run.iloc[0].run_id
        data = client.get_metric_history(id, metric)
        res.append([x.value for x in data])
    return np.array(res)


def plot(experiments, metric, args):
    for e in experiments:
        res = get_results(e, metric, args["trials"])
        met_means, acc_stds = np.mean(res, axis=0), np.std(res, axis=0)
        epochs = np.arange(1, args["epochs"] + 1)
        plt.plot(epochs, met_means, label=e)
        plt.fill_between(epochs, met_means - acc_stds, met_means + acc_stds, alpha=0.2)
    plt.ticklabel_format(style="sci", axis="y", scilimits=(-2, 2))
    plt.xlabel("Epochs")
    plt.ylabel(f"{metric}")
    plt.title("Experiment #1 vs Experiment #2")
    plt.legend()
    plt.grid()
    plt.savefig(f"results/{metric}_plot")
    plt.clf()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_file", "-yml", type=str, default=None)
    parser.add_argument("--batch_size", "-bs", type=int, default=64)
    parser.add_argument("--val_batch_size", "-vbs", type=int, default=1000)
    parser.add_argument("--epochs", "-e", type=int, default=5)
    parser.add_argument("--trials", "-t", type=int, default=1)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = vars(parse_arguments())

    # Plot metrics
    experiments = ["exp1", "exp2"]
    plot(experiments, "val_acc", args)
    plot(experiments, "val_loss", args)

    # Load and test model
    # _, testloader = load_data(**args)
    # model_path = f"models/1/{ID}/artifacts/model/data/model.pth"
    # model = torch.load(model_path, weights_only=False)
    # test(model, testloader)
    plt.show()
