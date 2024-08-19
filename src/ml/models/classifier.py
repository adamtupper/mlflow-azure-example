"""Define a PyTorch Lightning module for training a multi-class classification model."""

import lightning as L
import matplotlib.pyplot as plt
import torch
from torchmetrics.classification import Accuracy, ConfusionMatrix, AveragePrecision


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
        self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_map = AveragePrecision(task="multiclass", num_classes=self.num_classes)

        # Complex metrics
        self.confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=self.num_classes)

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
        self.log("train_acc_step", self.train_acc)
        self.log("train_ap_step", self.train_map)

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
        self.confusion_matrix(y_hat, y)

        return loss

    def on_validation_epoch_end(self):
        # Log scalar metrics
        self.log("val_acc_epoch", self.val_acc)
        self.log("val_ap_epoch", self.val_map)

        # Log confusion matrix
        fig, _ = self.confusion_matrix.plot()
        self.logger.experiment.log_figure(self.logger.run_id, fig, "confusion_matrix.png")
        plt.close()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
