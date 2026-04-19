import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchmetrics import Accuracy, Precision, Recall, JaccardIndex
from torchmetrics.segmentation import DiceScore
from DeepGeol.deepgeol.unet import UNet


class NPYDataset(Dataset):
    def __init__(self, data_path, mask_path):
        self.data  = np.load(data_path)
        self.masks = np.load(mask_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx]).float().permute(2, 0, 1)

        y = self.masks[idx]
        y = torch.tensor(y).float()
        y = y.permute(2, 0, 1) if y.ndim == 3 else y.unsqueeze(0)

        return x, y


def compute_metrics(loader, model, device):
    metrics = {
        "Pixel Accuracy": Accuracy(task="binary").to(device),
        "Précision"     : Precision(task="binary").to(device),
        "Rappel"        : Recall(task="binary").to(device),
        "Dice Score"    : DiceScore(num_classes=2, average="micro").to(device),
        "IoU"           : JaccardIndex(task="binary").to(device),
    }

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            pred_probs = torch.sigmoid(pred)
            pred_bin   = (pred_probs > 0.6).long()
            y_bin      = y.long()

            for metric in metrics.values():
                metric.update(pred_bin, y_bin)

    results = {}
    for name, metric in metrics.items():
        results[name] = metric.compute().item()

    return results


def evaluate():
    test_data_path = "/lium/buster1/larcher/M2/deep_learning/TP_CNN_UNet/data/test_data.npy"
    test_mask_path = "/lium/buster1/larcher/M2/deep_learning/TP_CNN_UNet/data/test_masks.npy"
    model_path     = "/info/raid-etu/m1/s2506992/Archive/checkpoints/best_model_v2.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    dataset = NPYDataset(test_data_path, test_mask_path)

    # split 80 / 20
    dev_size = int(0.8 * len(dataset))
    test_size = len(dataset) - dev_size

    dev_set, test_set = random_split(dataset, [dev_size, test_size])

    dev_loader  = DataLoader(dev_set, batch_size=4)
    test_loader = DataLoader(test_set, batch_size=4)

    # Model
    model = UNet(input_channels=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("Modele chargé.\n")

    # DEV
    dev_results = compute_metrics(dev_loader, model, device)
    print("=== Résultats DEV ===")
    for name, score in dev_results.items():
        print(f"  {name:<18}: {score * 100:.2f} %")

    print()

    # TEST
    test_results = compute_metrics(test_loader, model, device)
    print("=== Résultats TEST ===")
    for name, score in test_results.items():
        print(f"  {name:}: {score * 100:.2f} %")


if __name__ == "__main__":