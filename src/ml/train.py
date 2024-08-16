"""Fine-tune a ResNet-18 ImageNet model on the RetinaMNIST dataset.

This example uses Hydra to manage the configuration. For information on Hydra, see https://hydra.cc/.

Usage:

    python train.py
        dataset_dir=~/Downloads
        mlflow_tracking_uri=$(az ml workspace show --query mlflow_tracking_uri)
"""

import hydra
import lightning as L
import medmnist
import torch
import torchvision.transforms.v2 as transforms
from lightning.pytorch.loggers import MLFlowLogger
from models.classifier import Classifier
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

NUM_CLASSES = 5


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Define the training and validation transforms
    transform_train = transforms.Compose(
        [
            transforms.RandAugment(),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )
    transform_val = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )

    # Load the training and validation data
    train_dataset = medmnist.RetinaMNIST(
        split="train", download=True, size=224, transform=transform_train, root=cfg.dataset_dir
    )
    val_dataset = medmnist.RetinaMNIST(
        split="val", download=True, size=224, transform=transform_val, root=cfg.dataset_dir
    )

    # Load the pre-trained ResNet-18 model (modify the output layer to match the number of classes in the RetinaMNIST
    # dataset)
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

    mlf_logger = MLFlowLogger(experiment_name="lightning_logs", tracking_uri=cfg.mlflow_tracking_uri)
    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        fast_dev_run=cfg.fast_dev_run,
        logger=mlf_logger,
        log_every_n_steps=25,
    )
    trainer.fit(classifier, train_loader, val_loader)


if __name__ == "__main__":
    main()
