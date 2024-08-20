"""Evaluate a ResNet-18 model on the RetinaMNIST dataset."""

import hydra
import lightning as L
import medmnist
import mlflow
import torch
import torchvision.transforms.v2 as transforms
from models.classifier import Classifier
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchvision.models import resnet18

NUM_CLASSES = 5


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Setup MLflow
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment)

    # Define the test transforms
    transform_test = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )

    # Load the test data
    test_dataset = medmnist.RetinaMNIST(
        split="test", download=True, size=224, transform=transform_test, root=cfg.dataset_dir
    )
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, num_workers=cfg.dataloader_workers)

    # Load the model checkpoint (modify the output layer to match the number of classes in the RetinaMNIST dataset)
    model = resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    classifier = mlflow.pytorch.load_checkpoint(Classifier, cfg.mlflow_run_id, kwargs={"model": model, "lr": cfg.lr})

    # Evaluate the model on the test set
    trainer = L.Trainer()

    mlflow.pytorch.autolog()
    with mlflow.start_run(run_id=cfg.mlflow_run_id) as run:  # noqa: F841
        metrics = trainer.test(classifier, test_dataloader)[0]
        mlflow.log_metrics(metrics)

        if cfg.mlflow_register_model:
            result = mlflow.register_model(f"runs:/{cfg.mlflow_run_id}/model", cfg.mlflow_register_model)
            print(result)


if __name__ == "__main__":
    main()
