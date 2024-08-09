"""Fine-tune a ResNet-18 ImageNet model on the Oxford-IIIT Pet dataset.

Usage:

TODO: Add usage example.
"""

import hydra
import lightning as L
import torch
import torchvision.datasets as datasets
import torchvision.transforms.v2 as transforms
from data.subset_dataset import SubsetDataset
from models.classifier import Classifier
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, random_split
from torchvision.models import ResNet18_Weights, resnet18

NUM_CLASSES = 37


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Load the training and validation data
    train_val_dataset = datasets.OxfordIIITPet(
        root=cfg.dataset_dir, split="trainval", download=True
    )

    # Define the training and validation transforms
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    transform_val = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Split the data into training (80%) and validation (20%) sets
    train_size = int(0.8 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_subset, val_subset = random_split(train_val_dataset, [train_size, val_size])
    train_dataset = SubsetDataset(train_subset, transform=transform_train)
    val_dataset = SubsetDataset(val_subset, transform=transform_val)

    # Load the pre-trained ResNet-18 model (modify the output layer to match the number of classes in the Oxford-IIIT
    # Pet dataset)
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    classifier = Classifier(model, lr=cfg.lr)

    # Fine-tune the model on the training set
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.dataloader_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.dataloader_workers,
    )

    trainer = L.Trainer(max_epochs=cfg.max_epochs, fast_dev_run=cfg.fast_dev_run)
    trainer.fit(classifier, train_loader, val_loader)


if __name__ == "__main__":
    main()
