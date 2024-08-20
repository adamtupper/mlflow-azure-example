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
import mlflow
import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from lightning.pytorch.callbacks import ModelCheckpoint
from models.classifier import Classifier
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import ResNet18_Weights, resnet18

NUM_CLASSES = 5


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment)

    L.seed_everything(cfg.seed)

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

    # Weighted random sampling
    train_labels = train_dataset.labels.flatten()
    if cfg.inverse_weighted_sampling:
        class_sample_count = np.array([len(np.where(train_labels == t)[0]) for t in np.unique(train_labels)])
        class_weights = 1.0 / class_sample_count
    else:
        class_weights = np.ones_like(np.unique(train_labels))

    train_sample_weights = np.array([class_weights[t] for t in train_labels])
    train_sample_weights = torch.from_numpy(train_sample_weights).double()

    # Load the pre-trained ResNet-18 model (modify the output layer to match the number of classes in the RetinaMNIST
    # dataset)
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    classifier = Classifier(model, lr=cfg.lr)

    # Fine-tune the model on the training set
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.dataloader_workers,
        sampler=WeightedRandomSampler(train_sample_weights, len(train_sample_weights)),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.dataloader_workers,
    )

    # Train the model
    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        fast_dev_run=cfg.fast_dev_run,
        log_every_n_steps=10,
        callbacks=[
            ModelCheckpoint(
                monitor="val_map_epoch",
                mode="max",
                filename="{epoch}-{step}-{val_loss:.2f}-{val_map_epoch:.2f}",
                save_weights_only=True,
            )
        ],
    )

    mlflow.pytorch.autolog(log_every_n_step=10, checkpoint=True)
    with mlflow.start_run(log_system_metrics=True) as run:  # noqa: F841
        mlflow.log_params(OmegaConf.to_container(cfg))
        mlflow.log_params({f"class_weight_{i}": w for i, w in enumerate(class_weights)})

        trainer.fit(classifier, train_loader, val_loader)


if __name__ == "__main__":
    main()
