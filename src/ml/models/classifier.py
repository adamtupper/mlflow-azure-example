"""Define a PyTorch Lightning module for training a multi-class classification model."""

import lightning as L
import matplotlib.pyplot as plt
import mlflow
import torch
from torchmetrics.classification import AUROC, Accuracy, AveragePrecision, ConfusionMatrix, PrecisionRecallCurve
from torchvision.utils import make_grid


class Classifier(L.LightningModule):
    def __init__(self, model: torch.nn.Module, lr: float = 0.001):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = torch.nn.CrossEntropyLoss()
        self.num_classes = model.fc.out_features

        # Scalar metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.train_map = AveragePrecision(task="multiclass", num_classes=self.num_classes)  # Mean avg. precision (mAP)
        self.train_auroc = AUROC(task="multiclass", num_classes=self.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_map = AveragePrecision(task="multiclass", num_classes=self.num_classes)
        self.val_auroc = AUROC(task="multiclass", num_classes=self.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_map = AveragePrecision(task="multiclass", num_classes=self.num_classes)
        self.test_auroc = AUROC(task="multiclass", num_classes=self.num_classes)

        # Complex metrics (validation only)
        self.val_pr_curves = PrecisionRecallCurve(task="multiclass", num_classes=self.num_classes)
        self.val_confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=self.num_classes)
        self.test_pr_curves = PrecisionRecallCurve(task="multiclass", num_classes=self.num_classes)
        self.test_confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y = y.squeeze().long()
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        self.log("train_loss", loss)
        self.train_acc(y_hat, y)
        self.train_map(y_hat, y)
        self.train_auroc(y_hat, y)
        self.log("train_acc_step", self.train_acc)
        self.log("train_ap_step", self.train_map)
        self.log("train_auroc_step", self.train_auroc)

        if self.current_epoch == 0 and batch_idx == 0:
            # Log the first batch of images
            self._log_examples(x, prefix="train_")

        return loss

    def on_train_epoch_end(self):
        self.log("train_acc_epoch", self.train_acc)
        self.log("train_ap_epoch", self.train_map)

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y = y.squeeze().long()
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss)

        self.val_acc(y_hat, y)
        self.val_map(y_hat, y)
        self.val_auroc(y_hat, y)
        self.val_confusion_matrix(y_hat, y)
        self.val_pr_curves(y_hat, y)

        if self.current_epoch == 0 and batch_idx == 0:
            # Log the first batch of images
            self._log_examples(x, prefix="val_")

        return loss

    def on_validation_epoch_end(self):
        # Log scalar metrics
        self.log("val_acc_epoch", self.val_acc)
        self.log("val_map_epoch", self.val_map)
        self.log("val_auroc_epoch", self.val_auroc)

        # Log confusion matrix
        fig, _ = self.val_confusion_matrix.plot()
        mlflow.log_figure(fig, "val_confusion_matrix.png")
        plt.close()

        # Log precision-recall curve
        fig, _ = self.val_pr_curves.plot()
        mlflow.log_figure(fig, "val_pr_curves.png")
        plt.close()

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y = y.squeeze().long()
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        self.test_acc(y_hat, y)
        self.test_map(y_hat, y)
        self.test_auroc(y_hat, y)
        self.test_confusion_matrix(y_hat, y)
        self.test_pr_curves(y_hat, y)

        return loss

    def on_test_epoch_end(self):
        # Log scalar metrics
        self.log("test_acc_epoch", self.test_acc)
        self.log("test_map_epoch", self.test_map)
        self.log("test_auroc_epoch", self.test_auroc)

        # Log confusion matrix
        fig, _ = self.test_confusion_matrix.plot()
        mlflow.log_figure(fig, "test_confusion_matrix.png")
        plt.close()

        # Log precision-recall curve
        fig, _ = self.test_pr_curves.plot()
        mlflow.log_figure(fig, "test_pr_curves.png")
        plt.close()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def _log_examples(self, x: torch.Tensor, prefix: str = "") -> None:
        fig = plt.figure(figsize=(6.4, 6.4))
        plt.imshow(make_grid(x).cpu().numpy().transpose(1, 2, 0))
        plt.axis("off")
        mlflow.log_figure(fig, f"{prefix}examples.png")
        plt.close()
