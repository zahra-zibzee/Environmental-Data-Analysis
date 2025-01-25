import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from tqdm import tqdm
from unet_model import UNet


class ImagesDataset(Dataset):
    def __init__(
        self,
        images_folder: str | os.PathLike,
        x_transform: transforms.Compose = None,
        y_transform: transforms.Compose = None,
        extend_radius: int = 20,
    ):
        self.images_folder = Path(images_folder)
        self.x_transform = x_transform
        self.y_transform = y_transform
        self.images = list(self.images_folder.glob("depth*.png"))
        self.masks = list(self.images_folder.glob("mask*.npy"))
        assert (
            len(self.images) == len(self.masks) != 0
        ), f"Mismatch between images and masks, found {len(self.images)} images and {len(self.masks)} masks"
        self.images.sort()
        self.masks.sort()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_name = self.images[idx]
        mask_name = self.masks[idx]

        assert image_name.stem.replace("depth", "") == mask_name.stem.replace(
            "mask", ""
        ), f"Image and mask names do not match: {image_name.stem} and {mask_name.stem}"
        x = cv2.imread(str(self.images[idx]), cv2.IMREAD_GRAYSCALE)
        y = np.load(self.masks[idx])
        ones_locations = np.array(np.where(y == 1)).T
        y_shape = np.array(y.shape) - 1
        for delta_x in [-1, 0, 1]:
            for delta_y in [-1, 0, 1]:
                new_ones_locations = np.clip(
                    ones_locations + np.array([delta_x, delta_y]), 0, y_shape
                )

                y[new_ones_locations[:, 0], new_ones_locations[:, 1]] = 1

        y = (
            F.one_hot(torch.tensor(y).long(), num_classes=2)
            # .permute(2, 0, 1)
            .float().numpy()
        )
        if self.x_transform:
            x = self.x_transform(x)

        if self.y_transform:
            y = self.y_transform(y)
        return x, y


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    epoch_loss = 0.0
    for x, y in tqdm(dataloader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)


def validate(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device
) -> float:
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            val_loss += loss.item()
    return val_loss / len(dataloader)


def setup_model() -> tuple[nn.Module, optim.Optimizer, nn.Module, torch.device]:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_channels=1, n_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to(device))
    return model, optimizer, criterion, device


def create_dataset(data_folder: str | os.PathLike, batch_size: int):
    x_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
    )

    dataset = ImagesDataset(
        data_folder, x_transform=x_transform, y_transform=transforms.ToTensor()
    )
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train_model(
    data_folder: str | os.PathLike,
    model_save_path="unet_model.pth",
    epochs=10,
    batch_size=16,
):
    train_loader, val_loader = create_dataset(data_folder, batch_size)

    model, optimizer, criterion, device = setup_model()

    # Training loop
    all_train_loss, all_val_loss = [], []
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        all_train_loss.append(train_loss)
        all_val_loss.append(val_loss)
        print(
            f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

    # Save the trained model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    np.save("train_val_loss.npy", np.array([all_train_loss, all_val_loss]))
    print("Train and validation loss saved to train_val_loss.npy")
    return all_train_loss, all_val_loss, model, train_loader, val_loader


if __name__ == "__main__":
    # Example data (replace with your actual data)
    train_model("./data/patches", epochs=10)
